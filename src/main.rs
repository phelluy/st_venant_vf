// time RAYON_NUM_THREADS=8 cargo run --release

// file utilities
use std::fs::File;
use std::io::BufWriter;
use std::io::{Error, Write};

// parallel iterators
use rayon::prelude::*;

// majorant de la vitesse d'onde max
//const C: f64 = 15.;

// st venant: nombre de variables
const M: usize = 4;
// burgers
//const M: usize = 1;

const XMIN: f64 = -10.;
const XMAX: f64 = 10.;

const G: f64 = 9.81;

fn z(a: f64, b: f64) -> f64 {
    let sqrt = f64::sqrt;
    if a > b {
        2. * sqrt(G) / (sqrt(a) + sqrt(b))
    } else {
        sqrt(G) * sqrt((a + b) / 2. / a / b)
    }
}

fn dz(a: f64, b: f64) -> f64 {
    let sqrt = f64::sqrt;
    let pow = f64::powf;
    if a > b {
        -sqrt(G) * pow(sqrt(a) + sqrt(b), -0.2e1) * pow(b, -0.1e1 / 0.2e1)
    } else {
        sqrt(G)
            * sqrt(0.2e1)
            * pow((a + b) / a / b, -0.1e1 / 0.2e1)
            * (0.1e1 / a / b - (a + b) / a * pow(b, -0.2e1))
            / 0.4e1
    }
}

fn k(hl: f64, ul: f64, hr: f64, ur: f64, h: f64) -> f64 {
    ul - (h - hl) * z(hl, h) - ur - (h - hr) * z(hr, h)
}

fn dk(hl: f64, hr: f64, h: f64) -> f64 {
    -z(hl, h) - (h - hl) * dz(hl, h) - z(hr, h) - (h - hr) * dz(hr, h)
}

// riemann solver st venant
fn riemann(wl: [f64; 2], wr: [f64; 2], xi: f64) -> [f64; 2] {
    let hl = wl[0];
    let ul = wl[1] / hl;
    let hr = wr[0];
    let ur = wr[1] / hr;

    // méthode de Newton
    let mut hs = 1e-6;
    let mut dh: f64 = 1.;
    let mut iter = 0;
    while dh.abs() > 1e-10 {
        dh = -k(hl, ul, hr, ur, hs) / dk(hl, hr, hs);
        hs += dh;
        iter += 1;
        // println!(
        //     "iter ={} hl={} ul={} hr={} ur={} hs={} dh={}",
        //     iter, hl, ul, hr, ur, hs, dh
        // );
        if iter > 20 {
            panic!();
        }
    }
    let sqrt = f64::sqrt;

    let us = ul - (hs - hl) * z(hl, hs);
    let (lambda1m, lambda1p) = if hs < hl {
        (ul - sqrt(G * hl), us - sqrt(G * hs))
    } else {
        let j = -sqrt(G) * sqrt(hl * hs) * sqrt((hl + hs) / 2.);
        (ul + j / hl, ul + j / hl)
    };
    let (lambda2m, lambda2p) = if hs < hr {
        (us + sqrt(G * hs), ur + sqrt(G * hr))
    } else {
        let j = sqrt(G) * sqrt(hr * hs) * sqrt((hr + hs) / 2.);
        (ur + j / hr, ur + j / hr)
    };
    // println!("lambda1m={} lambda1p={}", lambda1m, lambda1p);
    // println!("lambda2m={} lambda2p={}", lambda2m, lambda2p);
    // panic!();
    let (h, u) = if xi < lambda1m {
        (hl, ul)
    } else if xi < lambda1p {
        let u1 = (ul + 2. * sqrt(G * hl) + 2. * xi) / 3.;
        ((u1 - xi) * (u1 - xi) / G, u1)
    } else if xi < lambda2m {
        (hs, us)
    } else if xi < lambda2p {
        let u2 = (ur - 2. * sqrt(G * hr) + 2. * xi) / 3.;
        ((u2 - xi) * (u2 - xi) / G, u2)
    } else {
        (hr, ur)
    };
    [h, h * u]
}

const R_AIR: f64 = 1.;
const R_WATER: f64 = 1000.;
const CSON: f64 = 10.;
const P0: f64 = 1e5;

fn rphi(phi: f64) -> f64 {
    phi * R_AIR + (1. - phi) * R_WATER
}

fn pres(r: f64, phi: f64) -> f64 {
    P0 + CSON * CSON * (r - rphi(phi))
}

fn prim2bal(y: [f64; 4]) -> [f64; 4] {
    let (p, u, v, phi) = (y[0], y[1], y[2], y[3]);

    let r0 = rphi(phi);
    let r = (p - P0) / CSON / CSON + r0;
    [r, r * u, r * v, r * phi]
}

fn bal2prim(w: [f64; 4]) -> [f64; 4] {
    let r = w[0];
    let (u, v, phi) = (w[1] / r, w[2] / r, w[3] / r);
    let p = pres(r, phi);
    [p, u, v, phi]
}

fn z_isot(ra: f64, rb: f64) -> f64 {
    let ln = f64::ln;
    let pow = f64::powf;

    if ra < rb {
        CSON * (ln(rb) - ln(ra)) / (rb - ra)
    } else {
        CSON * pow(ra * rb, -0.1e1 / 0.2e1)
    }
}

fn dz_isot(ra: f64, rb: f64) -> f64 {
    let ln = f64::ln;
    let pow = f64::powf;

    if ra < rb {
        -CSON / ra / (rb - ra) + CSON * (ln(rb) - ln(ra)) * pow(rb - ra, -0.2e1)
    } else {
        -CSON * pow(ra * rb, -0.3e1 / 0.2e1) * rb / 0.2e1
    }
}

fn f_isot(
    pl: f64,
    ul: f64,
    _vl: f64,
    phil: f64,
    pr: f64,
    ur: f64,
    _vr: f64,
    phir: f64,
    p: f64,
) -> f64 {
    let pow = f64::powf;
    ur - (pr - p)
        * pow(CSON, -0.2e1)
        * z_isot(
            (p - P0) * pow(CSON, -0.2e1) + rphi(phir),
            (pr - P0) * pow(CSON, -0.2e1) + rphi(phir),
        )
        - ul
        - (pl - p)
            * pow(CSON, -0.2e1)
            * z_isot(
                (p - P0) * pow(CSON, -0.2e1) + rphi(phil),
                (pl - P0) * pow(CSON, -0.2e1) + rphi(phil),
            )
}

fn df_isot(
    pl: f64,
    _ul: f64,
    _vl: f64,
    phil: f64,
    pr: f64,
    _ur: f64,
    _vr: f64,
    phir: f64,
    p: f64,
) -> f64 {
    let pow = f64::powf;
    let cson = CSON;
    pow(cson, -0.2e1)
        * z_isot(
            (p - P0) * pow(cson, -0.2e1) + rphi(phir),
            (pr - P0) * pow(cson, -0.2e1) + rphi(phir),
        )
        - (pr - p)
            * pow(cson, -0.4e1)
            * dz_isot(
                (p - P0) * pow(cson, -0.2e1) + rphi(phir),
                (pr - P0) * pow(cson, -0.2e1) + rphi(phir),
            )
        + pow(cson, -0.2e1)
            * z_isot(
                (p - P0) * pow(cson, -0.2e1) + rphi(phil),
                (pl - P0) * pow(cson, -0.2e1) + rphi(phil),
            )
        - (pl - p)
            * pow(cson, -0.4e1)
            * dz_isot(
                (p - P0) * pow(cson, -0.2e1) + rphi(phil),
                (pl - P0) * pow(cson, -0.2e1) + rphi(phil),
            )
}

// riemann solver isothermal model
fn riemisot(wl: [f64; 4], wr: [f64; 4], xi: f64) -> [f64; 4] {
    let rl = wl[0];
    let ul = wl[1] / rl;
    let vl = wl[2] / rl;
    let phil = wl[3] / rl;
    let pl = pres(rl, phil);

    let rr = wr[0];
    let ur = wr[1] / rr;
    let vr = wr[2] / rr;
    let phir = wr[3] / rr;
    let pr = pres(rr, phir);

    // méthode de Newton
    let mut ps = pres(0.01, phil).max(pres(0.01, phir));
    let mut dp: f64 = 1.;
    let mut iter = 0;
    while dp.abs() > 1e-10 {
        let f = f_isot(pl, ul, vl, phil, pr, ur, vr, phir, ps);
        let df = df_isot(pl, ul, vl, phil, pr, ur, vr, phir, ps);
        dp = -f / df;
        ps += dp;
        iter += 1;
        if iter > 20 {
            panic!();
        }
    }
    let sqrt = f64::sqrt;
    let ra = (ps - P0) / CSON / CSON + rphi(phil);
    let rb = (ps - P0) / CSON / CSON + rphi(phir);

    let us = ul + (rl - ra) * z_isot(ra, rl);
    //let us = ur - (rr-rb)*z_isot(rb,rr);
    // println!("us={} usp={}",us,usp);

    let (lambda1m, lambda1p) = if ra < rl {
        (ul - CSON, us - CSON)
    } else {
        let lambda = ul - CSON * sqrt(ra / rl);
        (lambda, lambda)
    };
    //println!("lm={} lp={}",lambda1m,lambda1p);
    let (lambda2m, lambda2p) = if rb < rr {
        (us + CSON, ur + CSON)
    } else {
        let lambda = ur + CSON * sqrt(rb / rr);
        (lambda, us / CSON * sqrt(rr / rb))
    };
    // println!("lm={} lp={}",lambda2m,lambda2p);

    let (p, u, v, phi) = if xi < lambda1m {
        (pl, ul, vl, phil)
    } else if xi < lambda1p {
        let u1 = xi + CSON;
        let r1 = rl * ((ul - u1) / CSON).exp();
        (pres(r1, phil), u1, vl, phil)
    } else if xi < us {
        (ps, us, vl, phil)
    } else if xi < lambda2m {
        (ps, us, vr, phir)
    } else if xi < lambda2p {
        let u2 = xi - CSON;
        let r2 = rr * ((u2 - ur) / CSON).exp();
        (pres(r2, phir), u2, vr, phir)
    } else {
        (pr, ur, vr, phir)
    };
    prim2bal([p, u, v, phi])
}

// burgers
// fn riemann(ul: [f64; M], ur: [f64; M], xi: f64) -> [f64; M] {
//     let ul = ul[0];
//     let ur = ur[0];
//     let u = if ul > ur {
//         let sigma = (ul + ur) / 2.;
//         if xi > sigma {
//             ur
//         } else {
//             ul
//         }
//     } else {
//         if xi <= ul {
//             ul
//         } else if xi >= ur {
//             ur
//         } else {
//             xi
//         }
//     };
//     [u]
// }

fn sol_exacte(x: f64, t: f64) -> [f64; M] {
    let pl = 1.1e5;
    let ul = 0.;
    let vl = 0.;
    let phil = 1.;

    let pr = 1e5;
    let ur = 0.;
    let vr = 0.;
    let phir = 1.;

    let yl = [pl, ul, vl, phil];
    let yr = [pr, ur, vr, phir];

    let wl = prim2bal(yl);
    let wr = prim2bal(yr);

    riemisot(wl, wr, x / (t + 1e-12))
}

// burgers
// fn sol_exacte(x: f64, t: f64) -> [f64; M] {
//     //let mut w = 0.;

//     let u = if t < 1. {
//         if x >= 1. {
//             0.
//         } else if x <= t {
//             1.
//         } else {
//             (1. - x) / (1. - t)
//         }
//     } else {
//         if x - 1. < (t - 1.) / 2. {
//             1.
//         } else {
//             0.
//         }
//     };

//     [u]

//     //w
// }

// fn fluxphy(u: [f64; M]) -> [f64; M] {
//     [u[0] * u[0] / 2.]
// }

// st venant
// fn fluxphy(w: [f64; 2]) -> [f64; 2] {
//     let h = w[0];
//     let u = w[1] / h;
//     [h * u, h * u * u + G * h * h / 2.]
// }

// bifluide isotherme
fn fluxphy(w: [f64; M]) -> [f64; M] {
    let y = bal2prim(w);
    let r = w[0];
    let (p, u, v, phi) = (y[0], y[1], y[2], y[3]);
    [r * u, r * u * u + p, r * u * v, r * u * phi]
}

// estimate of the time derivative from the space derivative
// d_t w = -f'(w) d_x w
fn jacob_dw(w: [f64; M], dw: [f64; M]) -> [f64; M] {
    let r = w[0];
    let u = w[1] / r;
    let v = w[2] / r;
    let phi = w[3] / r;

    let mut res = [0.; M];

    res[0] = dw[1];
    res[1] = (-u * u + CSON * CSON * (1. + R_AIR * phi / r - R_WATER * phi / r)) * dw[0]
        + 2. * u * dw[1]
        + CSON * CSON * (-R_AIR / r + R_WATER / r) * dw[3];
    res[2] = -u * v * dw[0] + u * dw[2] + v * dw[1];
    res[3] = -phi * u * dw[0] + phi * dw[1] + u * dw[3];
    res
}

fn fluxnum(wl: [f64; M], wr: [f64; M]) -> [f64; M] {
    // godunov
    let w = riemisot(wl, wr, 0.);
    fluxphy(w)
    // rusanov
    // let hl = wl[0];
    // let hr = wr[0];
    // let ul = wl[1] / hl;
    // let ur = wr[1] / hr;

    // let lambda = f64::max(ul.abs() + (G * hl).sqrt(), ur.abs() + (G * hr).sqrt());
    // let fl = fluxphy(wl);
    // let fr = fluxphy(wr);

    // let mut flux = [0.; M];
    // for i in 0..M {
    //     flux[i] = 0.5 * (fl[i] + fr[i]) - 0.5 * lambda * (wr[i] - wl[i]);
    // }
    // flux

    // vfroe
    // let hl = wl[0];
    // let hr = wr[0];
    // let ul = wl[1] / hl;
    // let ur = wr[1] / hr;

    // let hb = (hl + hr) / 2.;
    // let ub = (ul + ur) / 2.;

    // let cb = (G * hb).sqrt();

    // let lambda1 = ub - cb;
    // let lambda2 = ub + cb;

    // let ws = if lambda1 <= 0. && lambda2 <= 0. {
    //     wr
    // } else if lambda1 >= 0. && lambda2 >= 0. {
    //     wl
    // } else {
    //     let hs = hb - 0.5 / cb * hb * (ur - ul);
    //     let us = ub - 0.5 / cb * G * (hr - hl);
    //     [hs, hs * us]
    // };
    //fluxphy(ws)
}

fn minmod(a: f64, b: f64, c: f64) -> f64 {
    if (a < 0.) && (b < 0.) && (c < 0.) {
        a.max(b).max(c)
    } else if (a > 0.) && (b > 0.) && (c > 0.) {
        a.min(b).min(c)
    } else {
        0.
    }
}

fn main() -> Result<(), Error> {
    let nx = 400;

    let dx = (XMAX - XMIN) / nx as f64;

    println!("nx={} xmin={} xmax={} dx={}", nx, XMIN, XMAX, dx);


    // vector of cell centers
    let xi: Vec<f64> = (0..nx + 2)
        .map(|i| i as f64 * dx - dx / 2. + XMIN)
        .collect();

    // vector of solution at time n and n+1
    let mut wn: Vec<[f64; M]> = xi.iter().map(|x| sol_exacte(*x, 0.)).collect();
    let mut wnp1: Vec<[f64; M]> = wn.clone();

    // MUSCL space and time slopes
    let mut sn = vec![[0.;M]; nx + 2];
    let mut rn = vec![[0.;M]; nx + 2];

    let mut t = 0.;
    let tmax = 0.3;

    let cfl = 0.3;

    let mut iter_count = 0;

    while t < tmax {
        // calcul des pentes
        // construction d'un itérateur parallèle regroupant les vecteurs
        // wn , wn décalé à gauche, wn décalé à droite,
        // et les pentes en espace et en temps.
        let iterslope = wn
            .par_iter()
            .skip(1)
            .take(nx)
            .zip(wn.par_iter().take(nx).zip(wn.par_iter().skip(2).take(nx)))
            .zip(
                sn.par_iter_mut()
                    .skip(1)
                    .take(nx)
                    .zip(rn.par_iter_mut().skip(1).take(nx)),
            );

        // exécution de la boucle parallèle
        iterslope.for_each(|((w, (wm, wp)), (s, r))| {
            for k in 0..M {
                let a = (w[k] - wm[k]) / dx;
                let b = (wp[k] - w[k]) / dx;
                let c = (wp[k] - wm[k]) / 2. / dx;

                s[k] = minmod(a, b, c);
                //s[k] = 0.;
            }
            *r = jacob_dw(*w, *s);
        });

        let mut dt = 1e5;
        let mut umax = 0.;
        // calcul du pas de temps
        wn.iter().for_each(|w| {
            let uloc = w[1]/w[0];
            umax = (uloc + CSON).max(umax);
            dt = (cfl * dx / (uloc + CSON)).min(dt);
        });

        // construction d'un itérateur imbriqué balayant tous
        // les vecteurs nécessaires
        let w = wn.par_iter().skip(1).take(nx);
        let s = sn.par_iter().skip(1).take(nx);
        let r = rn.par_iter().skip(1).take(nx);

        let wsr = w.zip(s.zip(r));

        let wm = wn.par_iter().take(nx);
        let sm = sn.par_iter().take(nx);
        let rm = rn.par_iter().take(nx);

        let wmsr = wm.zip(sm.zip(rm));

        let wp = wn.par_iter().skip(2).take(nx);
        let sp = sn.par_iter().skip(2).take(nx);
        let rp = rn.par_iter().skip(2).take(nx);

        let wpsr = wp.zip(sp.zip(rp));

        let wnext = wnp1.par_iter_mut().skip(1).take(nx);

        let iter = wnext.zip(wsr.zip(wmsr.zip(wpsr)));

        // for (wp, (w, wd)) in iter {
        //     for k in 0..M {
        //         wp[k] = w[k] - dt / dx * C * (w[k] - wd[k]);
        //     }
        // }
        // exécution de la boucle parallèle
        // wnext: w au temps n+1
        // w , s ,r : w et les pentes au temps n
        // wm, sm, rm : idem mais décalé à gauche
        // wp, sp, rp : idem mais décalé à droite
        // en rust, |x| {f(x)} veut dire "la fonction qui à x associe f(x)"
        iter.for_each(|(wnext, ((w, (s, r)), ((wm, (sm, rm)), (wp, (sp, rp)))))| {
            // méthode MUSCL
            // flux à droite
            let mut wl = *w;
            let mut wr = *wp;
            for k in 0..M {
                wl[k] += s[k] * dx / 2. + r[k] * dt / 2.;
            }
            for k in 0..M {
                wr[k] += -sp[k] * dx / 2. + rp[k] * dt / 2.;
            }
            let flux = fluxnum(wl, wr); // flux(i, i+1)
            for k in 0..M {
                wnext[k] = w[k] - dt / dx * flux[k];
            }
            // flux à gauche
            let mut wl = *wm;
            let mut wr = *w;
            for k in 0..M {
                wl[k] += sm[k] * dx / 2. + rm[k] * dt / 2.;
            }
            for k in 0..M {
                wr[k] += -s[k] * dx / 2. + r[k] * dt / 2.;
            }
            let flux = fluxnum(wl, wr); // flux(i, i+1)
            for k in 0..M {
                wnext[k] += dt / dx * flux[k];
            }
        });
        //panic!();

        //for i in 1..nx + 1 {
        //    wnp1[i] = wn[i] - dt / dx * C * (wn[i] - wn[i - 1]);
        //}
        t += dt;
        iter_count += 1;
        // conditions aux limites
        wnp1[0] = sol_exacte(xi[0], t);
        wnp1[nx + 1] = sol_exacte(xi[nx + 1], t);

        // for (wn, wnp1) in wn.iter_mut().zip(wnp1.iter()) {
        //     *wn = *wnp1;
        // }
        // mise à jour
        wn.par_iter_mut()
            .zip(wnp1.par_iter())
            .for_each(|(wn, wnp1)| *wn = *wnp1);
        //println!("t={} dt={} vmax={}",t,dt,umax);
    }

    println!("Ok, {} iterations, final t ={}", iter_count, t);

    println!("plot rho");
    let ynum : Vec<f64> = wn.iter().map(|w| w[0]).collect();
    let yex: Vec<f64> = xi.iter().map(|x| sol_exacte(*x,t)[0]).collect();
    plot1d(&xi,&ynum, &yex);

    println!("plot u");
    let ynum : Vec<f64> = wn.iter().map(|w| w[1]/w[0]).collect();
    let yex: Vec<f64> = xi.iter().map(|x| {
        let w = sol_exacte(*x,t);
        w[1]/w[0]
    }).collect();
    plot1d(&xi,&ynum, &yex);

    println!("plot p");
    let ynum : Vec<f64> = wn.iter().map(|w| bal2prim(*w)[0]).collect();
    let yex: Vec<f64> = xi.iter().map(|x| {
        let w = sol_exacte(*x,t);
        bal2prim(w)[0]
    }).collect();
    plot1d(&xi,&ynum, &yex);

    // {
    //     let meshfile = File::create("trans.dat")?;
    //     let mut meshfile = BufWriter::new(meshfile); // create a buffer for faster writes...

    //     for i in 0..nx + 2 {
    //         let wex = sol_exacte(xi[i], t);
    //         let w = wn[i];
    //         let uex = wex[1] / wex[0];
    //         let u = w[1] / w[0];
    //         let hex = wex[0];
    //         let h = w[0];
    //         writeln!(meshfile, "{} {} {} {} {}", xi[i], h, hex, u, uex)?;
    //     }
    // } // ensures that the file is closed...

    // use std::process::Command;

    // Command::new("gnuplot")
    //     .arg("plotcom")
    //     .status()
    //     .expect("plot failed !");

    Ok(())
}

fn plot1d(x: &Vec<f64>, y: &Vec<f64>, z: &Vec<f64>) {
    let filename = "ploplo.dat";
    {
        let meshfile = File::create(filename).unwrap();
        let mut meshfile = BufWriter::new(meshfile); // create a buffer for faster writes...

        x.iter().zip(y.iter().zip(z.iter())).for_each(|(x, (y,z))| {
            writeln!(meshfile, "{} {} {}", *x, *y, *z).unwrap();
        });
    }

    use std::process::Command;
    Command::new("python3")
        .arg("src/plot1d.py")
        .status()
        .expect("plot failed !");
}
