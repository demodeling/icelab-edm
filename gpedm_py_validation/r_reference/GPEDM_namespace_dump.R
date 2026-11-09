

#### fitGP ####
fitGP <- function (data = NULL, y, x = NULL, pop = NULL, time = NULL, 
    E = NULL, tau = NULL, scaling = c("global", "local", "none"), 
    initpars = NULL, modeprior = 1, fixedpars = NULL, rhofixed = NULL, 
    rhomatrix = NULL, augdata = NULL, linprior = c("none", "local", 
        "global"), predictmethod = NULL, newdata = NULL, xnew = NULL, 
    popnew = NULL, timenew = NULL, ynew = NULL, returnGPgrad = FALSE, 
    exclradius = 0) 
{
    cl <- match.call()
    if ((!is.null(E) & is.null(tau)) | (is.null(E) & !is.null(tau))) {
        stop("Both E and tau must be supplied if generating lags internally.")
    }
    if (is.null(x) & (is.null(E) | is.null(tau))) {
        stop("x and/or E and tau must be supplied")
    }
    if (!is.null(rhofixed)) {
        if (!is.null(rhomatrix)) {
            stop("Supply either rhofixed or rhomatrix, not both")
        }
        if (rhofixed < 1e-04) {
            rhofixed = 1e-04
            message("rhofixed must between 0.0001 and 0.9999, setting to 0.0001")
        }
        if (rhofixed > 0.9999) {
            rhofixed = 0.9999
            message("rhofixed must between 0.0001 and 0.9999, setting to 0.9999")
        }
    }
    scaling <- match.arg(scaling)
    linprior <- match.arg(linprior)
    inputs = list()
    if (!is.null(colnames(x))) {
        inputs$x_names = colnames(x)
    }
    if (!is.null(data)) {
        inputs$y_names = y
        y = data[, y]
        if (!is.null(x)) {
            inputs$x_names = x
            x = data[, x]
        }
        if (!is.null(pop)) {
            inputs$pop_names = pop
            pop = data[, pop]
        }
        if (!is.null(time)) {
            inputs$time_names = time
            time = data[, time]
        }
    }
    if (is.null(pop)) {
        pop = rep(1, length(y))
    }
    if (is.null(time)) {
        up = unique(pop)
        time = y * 0
        for (i in 1:length(up)) {
            time[pop == up[i]] = 1:length(time[pop == up[i]])
        }
    }
    if (!is.null(E)) {
        if (is.null(x)) {
            x = y
            if (!is.null(inputs$y_names)) 
                inputs$x_names = inputs$y_names
        }
        x = makelags(y = x, pop = pop, E = E, tau = tau, yname = inputs$x_names)
        if (!is.null(inputs$x_names)) {
            inputs$x_names2 = colnames(x)
            if (any(grepl("_1", inputs$x_names) | grepl("_2", 
                inputs$x_names))) {
                message("It looks like x might already contain lags, are you sure you want to be using E and tau?")
            }
        }
        inputs$E = E
        inputs$tau = tau
    }
    x = as.matrix(x)
    d = ncol(x)
    if (!is.null(augdata)) {
        xaug = as.matrix(augdata[, inputs$x_names])
        yaug = augdata[, inputs$y_names]
        timeaug = augdata[, inputs$time_names]
        if (is.null(inputs$pop_names)) {
            popaug = rep(1, nrow(augdata))
        }
        else {
            popaug = augdata[, inputs$pop_names]
        }
        yd = c(y, yaug)
        xd = rbind(x, xaug)
        timed = c(time, timeaug)
        if (is.factor(pop)) {
            popd = factor(c(as.character(pop), as.character(popaug)), 
                levels = levels(pop))
        }
        else {
            popd = c(pop, popaug)
        }
        primary = c(rep(T, length(y)), rep(F, length(yaug)))
    }
    else {
        yd = y
        xd = x
        timed = time
        popd = pop
        primary = rep(T, length(y))
    }
    if (linprior == "none") {
        ydresid = yd
    }
    if (linprior == "global") {
        regdf = data.frame(y = yd, x = xd[, 1])
        linprior_reg = lm(y ~ x, data = regdf)
        ylinprior = yd * NA
        ylinprior[!is.na(xd[, 1])] = predict(linprior_reg)
        ydresid = yd - ylinprior
    }
    if (linprior == "local") {
        regdf = data.frame(y = yd, x = xd[, 1], pop = as.factor(popd))
        linprior_reg = lm(y ~ x * pop, data = regdf)
        ylinprior = yd * NA
        ylinprior[!is.na(xd[, 1])] = predict(linprior_reg)
        ydresid = yd - ylinprior
    }
    if (scaling == "none") {
        ymeans = mean(ydresid, na.rm = T)
        ysds = sd(ydresid, na.rm = T)
        yds = ydresid
        xmeans = apply(xd, 2, mean, na.rm = T)
        xsds = apply(xd, 2, sd, na.rm = T)
        xds = xd
        if (ymeans > 0.1 | ymeans < (-0.1) | ysds > 1.1 | ysds < 
            (-0.9)) {
            warning("y is not scaled properly, model may be unreliable ", 
                "mean=", round(ymeans, 3), " sd=", round(ysds, 
                  3), call. = F, immediate. = T)
        }
        if (any(xmeans > 0.1) | any(xmeans < (-0.1)) | any(xsds > 
            1.1) | any(xsds < (-0.9))) {
            warning("one or more x is not scaled properly, model may be unreliable. ", 
                "means=", paste(round(xmeans, 3), collapse = " "), 
                " sds=", paste(round(xsds, 3), collapse = " "), 
                call. = F, immediate. = T)
        }
    }
    if (scaling == "global") {
        ymeans = mean(ydresid, na.rm = T)
        ysds = sd(ydresid, na.rm = T)
        yds = scale(ydresid)
        xmeans = apply(xd, 2, mean, na.rm = T)
        xsds = apply(xd, 2, sd, na.rm = T)
        xds = apply(xd, 2, scale)
    }
    if (scaling == "local") {
        ymeans = tapply(ydresid, popd, mean, na.rm = T)
        ysds = tapply(ydresid, popd, sd, na.rm = T)
        xlist = split(as.data.frame(xd), popd)
        xmeans = lapply(xlist, function(x) apply(x, 2, mean, 
            na.rm = T))
        xsds = lapply(xlist, function(x) apply(x, 2, sd, na.rm = T))
        up = unique(popd)
        xds = xd
        yds = ydresid
        for (i in 1:length(up)) {
            locmean = ymeans[as.character(up[i]) == names(ymeans)]
            locsd = ysds[as.character(up[i]) == names(ysds)]
            yds[popd == up[i]] = (yds[popd == up[i]] - locmean)/locsd
            for (j in 1:d) {
                locmean = xmeans[[which(as.character(up[i]) == 
                  names(xmeans))]][j]
                locsd = xsds[[which(as.character(up[i]) == names(xsds))]][j]
                xds[popd == up[i], j] = (xd[popd == up[i], j] - 
                  locmean)/locsd
            }
        }
    }
    if (is.null(initpars)) {
        initpars = c(rep(0.1, d), 0.001, 1 - 0.001, 0.5)
    }
    if (is.null(fixedpars)) {
        fixedpars = rep(NA, times = d + 2)
    }
    if (length(fixedpars) != d + 2) {
        stop("Length of fixedpars must be number of predictors + 2 (phi, ve, sigma2)")
    }
    fixedpars_index = which(!is.na(fixedpars))
    initpars[fixedpars_index] = fixedpars[fixedpars_index]
    vemin = 1e-04
    sigma2min = 1e-04
    vemax = 4.9999
    sigma2max = 4.9999
    rhomin = 0
    rhomax = 1
    initparst = c(log(initpars[1:d]), logit(initpars[d + 1], 
        vemin, vemax), logit(initpars[d + 2], sigma2min, sigma2max), 
        logit(initpars[d + 3], rhomin, rhomax))
    if (length(unique(popd)) == 1) {
        initparst[d + 3] = 0
    }
    if (!is.null(rhomatrix)) {
        inputs$rhomatrix = rhomatrix
        up = unique(popd)
        if (is.null(colnames(rhomatrix)) | is.null(rownames(rhomatrix))) {
            colnames(rhomatrix) = up
            rownames(rhomatrix) = up
            message("rhomatrix entries assumed in order: ", paste(up, 
                collapse = " "), "\nName rows and columns to eliminate ambiguity.")
        }
        if (!all(as.character(up) %in% colnames(rhomatrix))) {
            stop("Population names are not all present in rhomatrix.")
        }
    }
    completerows = complete.cases(cbind(yds, xds))
    Y = yds[completerows]
    X = as.matrix(xds[completerows, , drop = FALSE])
    Pop = popd[completerows]
    Time = timed[completerows]
    Primary = primary[completerows]
    output = fmingrad_Rprop(initparst, Y, X, Pop, modeprior, 
        fixedpars, rhofixed, rhomatrix)
    ypred <- yfsd <- ysd <- yd * NA
    ypred[completerows] = output$mean
    yfsd[completerows] = sqrt(diag(output$cov))
    ysd[completerows] = sqrt(diag(output$cov) + output$pars["ve"])
    ypred = ypred[primary]
    yfsd = yfsd[primary]
    ysd = ysd[primary]
    if (scaling == "global") {
        ypred = ypred * ysds + ymeans
        yfsd = sqrt(yfsd^2 * ysds^2)
        ysd = sqrt(ysd^2 * ysds^2)
    }
    if (scaling == "local") {
        for (i in 1:length(up)) {
            locmean = ymeans[as.character(up[i]) == names(ymeans)]
            locsd = ysds[as.character(up[i]) == names(ysds)]
            ypred[pop == up[i]] = ypred[pop == up[i]] * locsd + 
                locmean
            yfsd[pop == up[i]] = sqrt(yfsd[pop == up[i]]^2 * 
                locsd^2)
            ysd[pop == up[i]] = sqrt(ysd[pop == up[i]]^2 * locsd^2)
        }
    }
    if (linprior != "none") {
        ypred = ypred + ylinprior
    }
    output$inputs = c(inputs, list(y = y, x = x, yd = yd, xd = xd, 
        pop = pop, time = time, completerows = completerows, 
        Y = Y, X = X, Pop = Pop, Time = Time, Primary = Primary))
    output$scaling = list(scaling = scaling, ymeans = ymeans, 
        ysds = ysds, xmeans = xmeans, xsds = xsds)
    if (linprior != "none") {
        output$linprior = list(linprior = linprior, linprior_reg = linprior_reg, 
            ylinprior = ylinprior)
    }
    output$insampresults = data.frame(timestep = time, pop = pop, 
        predmean = ypred, predfsd = yfsd, predsd = ysd, obs = y)
    output$insampfitstats = c(R2 = getR2(y, ypred), rmse = sqrt(mean((y - 
        ypred)^2, na.rm = T)), ln_post = -output$nllpost, ln_prior = output$lp, 
        lnL_LOO = output$lnL_LOO, df = output$df, SS = output$SS, 
        logdet = output$logdet)
    if (length(unique(pop)) > 1) {
        output$insampfitstatspop = getR2pop(y, ypred, pop)
        R2centered = getR2pop(y, ypred, pop, type = "centered")
        R2scaled = getR2pop(y, ypred, pop, type = "scaled")
        output$insampfitstats = c(output$insampfitstats, R2centered = R2centered, 
            R2scaled = R2scaled)
    }
    output[c("lnL_LOO", "df", "nllpost", "mean", "cov", "SS", 
        "logdet", "lp")] = NULL
    if (!is.null(predictmethod) | !is.null(xnew) | !is.null(newdata)) {
        predictresults = predict.GP(output, predictmethod, newdata, 
            xnew, popnew, timenew, ynew, returnGPgrad, exclradius)
        output = c(output, predictresults)
    }
    output$call = cl
    class(output) = c("GP", "GPpred")
    return(output)
}
<bytecode: 0x3142c6e8>
<environment: namespace:GPEDM>


#### fitGP_fish ####
fitGP_fish <- function (data, y, m, h, z = NULL, pop = NULL, time = NULL, scaling = c("global", 
    "local", "none"), initpars = NULL, modeprior = 1, fixedpars = NULL, 
    rhofixed = NULL, rhomatrix = NULL, augdata = NULL, linprior = "none", 
    predictmethod = NULL, newdata = NULL, xname = "escapement", 
    ytrans = c("none", "log", "gr1", "gr2"), bfixed = NULL, bshared = TRUE) 
{
    if (length(m) != length(h)) {
        stop("m and h must have the same number of columns")
    }
    cl <- match.call()
    scaling <- match.arg(scaling)
    ytrans <- match.arg(ytrans)
    md = data[, m, drop = F]
    hd = data[, h, drop = F]
    if (!is.null(bfixed)) {
        b = bfixed
        if (length(b) > 1) {
            if (is.null(pop)) {
                stop("pop must be supplied if multiple b values supplied")
            }
            popvec = data[, pop]
            np = length(unique(popvec))
            if (length(b) != np | !all(unique(popvec) %in% names(b))) {
                stop("bfixed must have length equal to number of pops and be named accordingly. Pop names: ", 
                  paste(unique(popvec), collapse = " "))
            }
        }
    }
    else {
        if (!is.null(pop)) {
            popvec = data[, pop]
            np = length(unique(popvec))
        }
        else {
            np = 1
        }
        if (bshared | np == 1) {
            bmax = min(md/hd, na.rm = T)
            boptim = optimize(GPnlike_fish, interval = c(0, bmax), 
                data = data, y = y, m = m, h = h, z = z, pop = pop, 
                time = time, scaling = scaling, initpars = initpars, 
                modeprior = modeprior, fixedpars = fixedpars, 
                rhofixed = rhofixed, rhomatrix = rhomatrix, augdata = augdata, 
                linprior = linprior, xname = xname, ytrans = ytrans)
            b = boptim$minimum
        }
        else {
            up = unique(popvec)
            bmax = vector(length = np)
            for (i in 1:np) {
                bmax[i] = min(md[popvec == up[i], ]/hd[popvec == 
                  up[i], ], na.rm = T)
            }
            parinits = bmax/2
            names(parinits) = up
            boptim = optim(parinits, GPnlike_fish, bmax = bmax, 
                data = data, y = y, m = m, h = h, z = z, pop = pop, 
                time = time, scaling = scaling, initpars = initpars, 
                modeprior = modeprior, fixedpars = fixedpars, 
                rhofixed = rhofixed, rhomatrix = rhomatrix, augdata = augdata, 
                linprior = linprior, xname = xname, ytrans = ytrans)
            b = boptim$par
            if (any(b < 0) | any(b > bmax)) {
                warning("one or more b estimates outside of reasonable bounds.\n", 
                  paste(up, collapse = " "), "\n", "bmax: ", 
                  paste(round(bmax, 5), collapse = " "), "\n", 
                  "b estimated: ", paste(round(b, 5), collapse = " "))
            }
        }
    }
    if (length(b) > 1) {
        popvec = data[, pop]
        bvec = b[match(popvec, names(b))]
    }
    else {
        bvec = rep(b, nrow(md))
    }
    x2 = md - bvec * hd
    x = paste(xname, 1:ncol(x2), sep = "_")
    colnames(x2) = x
    data = cbind(data, x2)
    yt = paste(y, "trans", sep = "_")
    yd = ytransfun(y = data[, y], m1 = md[, 1], e1 = x2[, 1], 
        ytrans = ytrans)
    yd = data.frame(yd)
    colnames(yd) = yt
    data = cbind(data, yd)
    if (!is.null(augdata)) {
        maug = augdata[, m, drop = F]
        haug = augdata[, h, drop = F]
        if (length(b) > 1) {
            popvecaug = augdata[, pop]
            bvecaug = b[match(popvecaug, names(b))]
        }
        else {
            bvecaug = rep(b, nrow(maug))
        }
        x2aug = maug - bvecaug * haug
        colnames(x2aug) = x
        augdata = cbind(augdata, x2aug)
        ydaug = ytransfun(y = augdata[, y], m1 = maug[, 1], e1 = x2aug[, 
            1], ytrans = ytrans)
        ydaug = data.frame(ydaug)
        colnames(ydaug) = yt
        augdata = cbind(augdata, ydaug)
    }
    if (!is.null(z)) {
        x = c(x, z)
    }
    output = fitGP(data = data, y = yt, x = x, pop = pop, time = time, 
        scaling = scaling, initpars = initpars, modeprior = modeprior, 
        fixedpars = fixedpars, rhofixed = rhofixed, rhomatrix = rhomatrix, 
        augdata = augdata, linprior = linprior)
    output$b = b
    ypred_trans = output$insampresults$predmean
    ypred = ytransfuninv(y = ypred_trans, m1 = md[, 1], e1 = x2[, 
        1], ytrans = ytrans)
    yobs = data[, y]
    colnames(output$insampresults) = c("timestep", "pop", "predmean_trans", 
        "predfsd_trans", "predsd_trans", "obs_trans")
    res2 = cbind.data.frame(predmean = ypred, obs = yobs, data[, 
        x, drop = F])
    output$insampresults = cbind.data.frame(output$insampresults, 
        res2)
    output$insampfitstats["R2"] = getR2(yobs, ypred)
    output$insampfitstats["rmse"] = sqrt(mean((yobs - ypred)^2, 
        na.rm = T))
    pop = output$insampresults$pop
    if (length(unique(pop)) > 1) {
        output$insampfitstatspop = getR2pop(yobs, ypred, pop)
        output$insampfitstats["R2centered"] = getR2pop(yobs, 
            ypred, pop, type = "centered")
        output$insampfitstats["R2scaled"] = getR2pop(yobs, ypred, 
            pop, type = "scaled")
    }
    output$inputs$m = md
    output$inputs$m_names = m
    output$inputs$h_names = h
    output$inputs$z_names = z
    output$inputs$y0_names = y
    output$inputs$ytrans = ytrans
    if (!is.null(predictmethod) | !is.null(newdata)) {
        predictresults = predict.GP(output, predictmethod, newdata)
        output = c(output, predictresults)
    }
    output$call = cl
    class(output) = c("GP", "GPpred")
    return(output)
}
<bytecode: 0x3155ace8>
<environment: namespace:GPEDM>


#### fmingrad_Rprop ####
fmingrad_Rprop <- function (initparst, Y, X, pop, modeprior, fixedpars, rhofixed, 
    rhomatrix) 
{
    p = length(initparst)
    Delta0 = 0.1 * rep(1, p)
    Deltamin = 1e-06
    Deltamax = 50
    eta_minus = 0.5
    eta_minus = eta_minus - 1
    eta_plus = 1.2
    eta_plus = eta_plus - 1
    maxiter = 200
    D = list()
    for (i in 1:ncol(X)) {
        D[[i]] = laGP::distance(X[, i])
    }
    output = getlikegrad(initparst, Y, X, pop, modeprior, fixedpars, 
        rhofixed, rhomatrix, D, returngradonly = T)
    nllpost = output$nllpost
    grad = output$grad
    s = drop(sqrt(grad %*% grad))
    pa = initparst
    iter = 0
    del = Delta0
    df = 10
    while (s > 1e-04 & iter < maxiter & df > 1e-07) {
        panew = pa - sign(grad) * del
        output_new = getlikegrad(panew, Y, X, pop, modeprior, 
            fixedpars, rhofixed, rhomatrix, D, returngradonly = T)
        nllpost_new = output_new$nllpost
        grad_new = output_new$grad
        s = drop(sqrt(grad_new %*% grad_new))
        df = abs(nllpost_new/nllpost - 1)
        gc = grad * grad_new
        tdel = del * (1 + eta_plus * (gc > 0) * 1 + eta_minus * 
            (gc < 0) * 1)
        del = ifelse(tdel < Deltamin, Deltamin, ifelse(tdel > 
            Deltamax, Deltamax, tdel))
        pa = panew
        grad = grad_new
        nllpost = nllpost_new
        iter = iter + 1
    }
    paopt = pa
    output_new = getlikegrad(paopt, Y, X, pop, modeprior, fixedpars, 
        rhofixed, rhomatrix, D, returngradonly = F)
    output_new$iter = iter
    return(output_new)
}
<bytecode: 0x3158e1c8>
<environment: namespace:GPEDM>


#### getconditionals ####
getconditionals <- function (fit, xrange = "default", extrap = 0.01, nvals = 25, 
    plot = T) 
{
    iKVs = fit$covm$iKVs
    phi = fit$pars[grepl("phi", names(fit$pars))]
    sigma2 = fit$pars[names(fit$pars) == "sigma2"]
    rho = fit$pars[names(fit$pars) == "rho"]
    rhomatrix = fit$inputs$rhomatrix
    X = fit$inputs$X
    Y = fit$inputs$Y
    Pop = fit$inputs$Pop
    scaling = fit$scaling$scaling
    ymeans = fit$scaling$ymeans
    ysds = fit$scaling$ysds
    xmeans = fit$scaling$xmeans
    xsds = fit$scaling$xsds
    if (xrange == "default") {
        if (scaling == "local") {
            xrange2 = "local"
        }
        else {
            xrange2 = "global"
        }
    }
    else {
        xrange2 = xrange
    }
    up = unique(Pop)
    np = length(up)
    d = ncol(X)
    Tslp = nvals
    outlist = NULL
    for (k in 1:np) {
        indi = which(Pop == up[k])
        xval <- predmean <- predsd <- xg <- matrix(0, nrow = Tslp, 
            ncol = d)
        poppred = rep(up[k], Tslp)
        for (i in 1:d) {
            if (xrange2 == "local") {
                xgextrap = (max(X[indi, i]) - min(X[indi, i])) * 
                  extrap
                xgi = seq(min(X[indi, i]) - xgextrap, max(X[indi, 
                  i]) + xgextrap, length.out = Tslp)
            }
            else {
                xgextrap = (max(X[, i]) - min(X[, i])) * extrap
                xgi = seq(min(X[, i]) - xgextrap, max(X[]) + 
                  xgextrap, length.out = Tslp)
            }
            xp = xg
            xp[, i] = xgi
            covmnew = getcov(phi, sigma2, rho, X, xp, Pop, poppred, 
                rhomatrix)
            Cs = covmnew$Cd
            predmean[, i] = Cs %*% (iKVs %*% Y)
            predvar = numeric(length = Tslp)
            for (j in 1:Tslp) {
                predvar[j] = sigma2 - Cs[j, ] %*% iKVs %*% Cs[j, 
                  ]
            }
            predsd[, i] = sqrt(predvar)
            xval[, i] = xgi
        }
        outlist[[k]] = list(poppred = poppred, xval = xval, predmean = predmean, 
            predsd = predsd)
    }
    if (scaling == "global") {
        for (i in 1:np) {
            for (j in 1:d) {
                outlist[[i]]$predmean[, j] = outlist[[i]]$predmean[, 
                  j] * ysds + ymeans
                outlist[[i]]$predsd[, j] = sqrt(outlist[[i]]$predsd[, 
                  j]^2 * ysds^2)
                outlist[[i]]$xval[, j] = outlist[[i]]$xval[, 
                  j] * xsds[j] + xmeans[j]
            }
        }
    }
    if (scaling == "local") {
        for (i in 1:np) {
            for (j in 1:d) {
                locmean = ymeans[as.character(up[i]) == names(ymeans)]
                locsd = ysds[as.character(up[i]) == names(ysds)]
                outlist[[i]]$predmean[, j] = outlist[[i]]$predmean[, 
                  j] * locsd + locmean
                outlist[[i]]$predsd[, j] = sqrt(outlist[[i]]$predsd[, 
                  j]^2 * locsd^2)
                locmean = xmeans[[which(as.character(up[i]) == 
                  names(xmeans))]][j]
                locsd = xsds[[which(as.character(up[i]) == names(xsds))]][j]
                outlist[[i]]$xval[, j] = outlist[[i]]$xval[, 
                  j] * locsd + locmean
            }
        }
    }
    if (!is.null(fit$linprior)) {
        for (i in 1:np) {
            regdf_new = data.frame(x = outlist[[i]]$xval[, 1], 
                pop = outlist[[i]]$poppred)
            ynewlinprior = predict(fit$linprior$linprior_reg, 
                newdata = regdf_new)
            outlist[[i]]$predmean[, 1] = outlist[[i]]$predmean[, 
                1] + ynewlinprior
            if (d > 1) {
                for (j in 2:d) {
                  if (scaling == "global") 
                    locmean1 = xmeans[1]
                  if (scaling == "local") 
                    locmean1 = xmeans[[which(as.character(up[i]) == 
                      names(xmeans))]][1]
                  if (scaling == "none") 
                    locmean1 = 0
                  regdf_new = data.frame(x = rep(locmean1, nvals), 
                    pop = outlist[[i]]$poppred)
                  ynewlinprior = predict(fit$linprior$linprior_reg, 
                    newdata = regdf_new)
                  outlist[[i]]$predmean[, j] = outlist[[i]]$predmean[, 
                    j] + ynewlinprior
                }
            }
        }
    }
    if (!is.null(fit$inputs$x_names2)) {
        xlabels = fit$inputs$x_names2
    }
    else if (!is.null(fit$inputs$x_names)) {
        xlabels = fit$inputs$x_names
    }
    else {
        xlabels = paste0("x", 1:d)
    }
    out = lapply(outlist, function(x) {
        cc2 = cbind.data.frame(x$poppred, x$xval, x$predmean, 
            x$predsd)
        colnames(cc2) = c("pop", xlabels, paste0(xlabels, "_yMean"), 
            paste0(xlabels, "_ySD"))
        return(cc2)
    })
    out = do.call("rbind", out)
    if (plot) {
        if (!is.null(fit$inputs$y_names)) {
            yl = fit$inputs$y_names
            if (!is.null(fit$b)) {
                if (fit$inputs$ytrans == "none") {
                  yl = sub("_trans", "", yl)
                }
            }
        }
        else {
            yl = "y"
        }
        old.par <- par(no.readonly = TRUE)
        on.exit(par(old.par), add = T, after = F)
        par(mfrow = c(min(4, np), min(4, d)), mar = c(5, 4, 2, 
            2))
        for (i in 1:np) {
            ylims = range(out[out$pop == up[i], grep("_yMean", 
                colnames(out))] + out[out$pop == up[i], grep("_ySD", 
                colnames(out))], out[out$pop == up[i], grep("_yMean", 
                colnames(out))] - out[out$pop == up[i], grep("_ySD", 
                colnames(out))])
            pdata = outlist[[i]]
            for (j in 1:d) {
                plot(pdata$xval[, j], pdata$predmean[, j], type = "l", 
                  xlab = xlabels[j], ylab = yl, main = up[i], 
                  ylim = ylims)
                lines(pdata$xval[, j], pdata$predmean[, j] + 
                  pdata$predsd[, j], lty = 2)
                lines(pdata$xval[, j], pdata$predmean[, j] - 
                  pdata$predsd[, j], lty = 2)
            }
        }
    }
    return(invisible(out))
}
<bytecode: 0x315c9020>
<environment: namespace:GPEDM>


#### getcov ####
getcov <- function (phi, sigma2, rho, X, X2, pop, pop2, rhomat = NULL) 
{
    Tsl = nrow(X)
    Tslp = nrow(X2)
    lC0 = -laGP::distance(t(t(X2) * sqrt(phi)), t(t(X) * sqrt(phi)))
    popsame = (outer(pop2, pop, "==")) * 1
    if (!is.null(rhomat)) {
        up = unique(pop)
        np = length(up)
        Rmat = matrix(1, nrow = Tslp, ncol = Tsl)
        for (i in 1:np) {
            for (j in 1:np) {
                indi = which(pop2 == up[i])
                indj = which(pop == up[j])
                Rmat[indi, indj] = rhomat[as.character(up[i]), 
                  as.character(up[j])]
            }
        }
        Cd = sigma2 * exp(lC0) * Rmat
    }
    else {
        Cd = sigma2 * exp(lC0) * (popsame + rho * (1 - popsame))
    }
    return(list(popsame = popsame, Cd = Cd))
}
<bytecode: 0x315f5878>
<environment: namespace:GPEDM>


#### getcovinv ####
getcovinv <- function (Sigma) 
{
    mm <- as(Sigma, "dpoMatrix")
    L = chol(mm)
    iKVs = chol2inv(L)
    return(list(L = as.matrix(L), iKVs = as.matrix(iKVs)))
}
<bytecode: 0x3161bb98>
<environment: namespace:GPEDM>


#### getGPgrad ####
getGPgrad <- function (phi, Xnew, X, Cdi, iKVs, Y) 
{
    grad = t(-2 * t(t(Xnew - t(X)) %*% diag(phi)) %*% (t(Cdi) * 
        iKVs %*% Y))
}
<bytecode: 0x3161e3e0>
<environment: namespace:GPEDM>


#### getlikegrad ####
getlikegrad <- function (parst, Y, X, pop, modeprior, fixedpars, rhofixed, rhomatrix, 
    D, returngradonly) 
{
    d = ncol(X)
    vemin = 1e-04
    sigma2min = 1e-04
    vemax = 4.9999
    sigma2max = 4.9999
    rhomin = 1e-04
    rhomax = 1
    phi = exp(parst[1:d])
    ve = (vemax - vemin)/(1 + exp(-parst[d + 1])) + vemin
    sigma2 = (sigma2max - sigma2min)/(1 + exp(-parst[d + 2])) + 
        sigma2min
    rho = (rhomax - rhomin)/(1 + exp(-parst[d + 3])) + rhomin
    if (length(unique(pop)) == 1 | !is.null(rhomatrix)) {
        rhofixed = 0.5
    }
    if (!is.null(rhofixed)) {
        rho = rhofixed
        parst[d + 3] = logit(rho, rhomin, rhomax)
    }
    if (any(!is.na(fixedpars[1:d]))) {
        phi[which(!is.na(fixedpars[1:d]))] = fixedpars[which(!is.na(fixedpars[1:d]))]
        parst[1:d] = log(phi)
    }
    if (!is.na(fixedpars[d + 1])) {
        ve = fixedpars[d + 1]
        parst[d + 1] = logit(ve, vemin, vemax)
    }
    if (!is.na(fixedpars[d + 2])) {
        sigma2 = fixedpars[d + 2]
        parst[d + 2] = logit(sigma2, sigma2min, sigma2max)
    }
    dpars = c(phi, (ve - vemin) * (1 - (ve - vemin)/(vemax - 
        vemin)), (sigma2 - sigma2min) * (1 - (sigma2 - sigma2min)/(sigma2max - 
        sigma2min)), (rho - rhomin) * (1 - (rho - rhomin)/(rhomax - 
        rhomin)))
    priors = getpriors(phi, ve, sigma2, rho, modeprior, sigma2max, 
        vemax)
    lp = priors$lp
    dlp = priors$dlp
    covm = getcov(phi, sigma2, rho, X, X, pop, pop, rhomatrix)
    Cd = covm$Cd
    Id = diag(ncol(Cd))
    Sigma = Cd + ve * Id
    covm$Sigma = Sigma
    Sigma = (Sigma + t(Sigma))/2
    icov = getcovinv(Sigma)
    iKVs = icov$iKVs
    L = icov$L
    covm$iKVs = iKVs
    a = iKVs %*% Y
    like = -0.5 * Y %*% a - sum(log(diag(L)))
    dl = 0 * dlp
    vQ = 0.5 * matrix2vector(a %*% t(a) - iKVs)
    for (i in 1:d) {
        if (is.na(fixedpars[i])) {
            dC = -multMat(D[[i]], Cd)
            vdC = matrix2vector(dC)
            dl[i] = 0.5 * innerProd(vQ, vdC)
        }
    }
    if (is.na(fixedpars[d + 1])) {
        vdC = matrix2vector(Id)
        dl[d + 1] = innerProd(vQ, vdC)
    }
    if (is.na(fixedpars[d + 2])) {
        vdC = matrix2vector(Cd/sigma2)
        dl[d + 2] = innerProd(vQ, vdC)
    }
    if (!is.null(rhofixed)) {
        dl[d + 3] = 0
    }
    else {
        popsame = covm$popsame
        dC = Cd * (1 - popsame)/rho
        vdC = matrix2vector(dC)
        dl[d + 3] = innerProd(vQ, vdC)
    }
    J = dl + dlp
    neglgrad = -J * dpars
    neglpost = -(like + lp)
    if (returngradonly) {
        return(list(nllpost = neglpost, grad = neglgrad))
    }
    SS = Y %*% a
    logdet = sum(log(diag(L)))
    pars = c(phi, ve, sigma2, rho)
    mpt = Cd %*% a
    Cdt = Cd - Cd %*% iKVs %*% Cd
    names(pars) = c(paste0("phi", 1:length(phi)), "ve", "sigma2", 
        "rho")
    names(parst) = c(paste0("phi", 1:length(phi)), "ve", "sigma2", 
        "rho")
    names(neglgrad) = c(paste0("phi", 1:length(phi)), "ve", "sigma2", 
        "rho")
    lnL_LOO = 0.5 * sum(log(diag(iKVs))) - 0.5 * sum(a^2/diag(iKVs))
    df = sum(diag((Cd %*% iKVs)))
    out = list(pars = pars, parst = parst, nllpost = neglpost, 
        grad = neglgrad, lnL_LOO = lnL_LOO, df = df, mean = mpt, 
        cov = Cdt, covm = covm, SS = SS, logdet = logdet, lp = lp)
    return(out)
}
<bytecode: 0x31624480>
<environment: namespace:GPEDM>


#### getpriors ####
getpriors <- function (phi, ve, sigma2, rho, modeprior, sigma2max, vemax) 
{
    lam_phi = (2^(modeprior - 1))^2 * pi/2
    lp_phi = -0.5 * sum(phi^2)/lam_phi
    dlp_phi = -(phi^1)/lam_phi
    a_sigma2 = 2
    b_sigma2 = 2
    lp_sigma2 = (a_sigma2 - 1) * log(sigma2/sigma2max) + (b_sigma2 - 
        1) * log(1 - sigma2/sigma2max)
    dlp_sigma2 = (a_sigma2 - 1)/(sigma2) - (b_sigma2 - 1)/(sigma2max - 
        sigma2)
    a_ve = 2
    b_ve = 2
    lp_ve = (a_ve - 1) * log(ve/vemax) + (b_ve - 1) * log(1 - 
        ve/vemax)
    dlp_ve = (a_ve - 1)/(ve) - (b_ve - 1)/(vemax - ve)
    a_rho = 2
    b_rho = 2
    lp_rho = (a_rho - 1) * log(rho) + (b_rho - 1) * log(1 - rho)
    dlp_rho = (a_rho - 1)/rho - (b_rho - 1)/(1 - rho)
    lp = lp_phi + lp_ve + lp_sigma2 + lp_rho
    dlp = c(dlp_phi, dlp_ve, dlp_sigma2, dlp_rho)
    return(list(lp = lp, dlp = dlp))
}
<bytecode: 0x3165ab18>
<environment: namespace:GPEDM>


#### getR2 ####
getR2 <- function (obs, pred) 
{
    d = na.omit(cbind(obs, pred))
    R2 = 1 - sum((d[, 1] - d[, 2])^2)/sum((d[, 1] - mean(d[, 
        1]))^2)
    return(R2)
}
<bytecode: 0x3166c8f8>
<environment: namespace:GPEDM>


#### getR2pop ####
getR2pop <- function (obs, pred, pop, type = c("within", "centered", "scaled")) 
{
    type = match.arg(type)
    up = unique(pop)
    np = length(up)
    if (type == "within") {
        R2pop <- rmsepop <- numeric(np)
        names(R2pop) = up
        names(rmsepop) = up
        for (p in 1:np) {
            ind = which(pop == up[p])
            R2pop[p] = getR2(obs[ind], pred[ind])
            rmsepop[p] = sqrt(mean((obs[ind] - pred[ind])^2, 
                na.rm = T))
        }
        insampfitstatspop = list(R2pop = R2pop, rmsepop = rmsepop)
        return(insampfitstatspop)
    }
    if (type == "centered") {
        ymeans = tapply(obs, pop, mean, na.rm = T)
        obs_s = obs
        pred_s = pred
        for (p in 1:length(up)) {
            locmean = ymeans[as.character(up[p]) == names(ymeans)]
            obs_s[pop == up[p]] = (obs_s[pop == up[p]] - locmean)
            pred_s[pop == up[p]] = (pred_s[pop == up[p]] - locmean)
        }
        R2 = getR2(obs_s, pred_s)
        return(R2)
    }
    if (type == "scaled") {
        ymeans = tapply(obs, pop, mean, na.rm = T)
        ysds = tapply(obs, pop, sd, na.rm = T)
        obs_s = obs
        pred_s = pred
        for (p in 1:length(up)) {
            locmean = ymeans[as.character(up[p]) == names(ymeans)]
            locsd = ysds[as.character(up[p]) == names(ysds)]
            obs_s[pop == up[p]] = (obs_s[pop == up[p]] - locmean)/locsd
            pred_s[pop == up[p]] = (pred_s[pop == up[p]] - locmean)/locsd
        }
        R2 = getR2(obs_s, pred_s)
        return(R2)
    }
}
<bytecode: 0x31676bb8>
<environment: namespace:GPEDM>


#### getrhomatrix ####
getrhomatrix <- function (data = NULL, y, x = NULL, pop, time = NULL, E = NULL, 
    tau = NULL, scaling = c("global", "local", "none"), initpars = NULL, 
    modeprior = 1, fixedpars = NULL, augdata = NULL) 
{
    scaling <- match.arg(scaling)
    inputs = list()
    if (!is.null(colnames(x))) {
        inputs$x_names = colnames(x)
    }
    if (!is.null(data)) {
        inputs$y_names = y
        y = data[, y]
        if (!is.null(x)) {
            inputs$x_names = x
            x = data[, x]
        }
        if (!is.null(pop)) {
            inputs$pop_names = pop
            pop = data[, pop]
        }
        if (!is.null(time)) {
            inputs$time_names = time
            time = data[, time]
        }
    }
    if (is.null(pop)) {
        pop = rep(1, length(y))
    }
    if (is.null(time)) {
        up = unique(pop)
        time = y * 0
        for (i in 1:length(up)) {
            time[pop == up[i]] = 1:length(time[pop == up[i]])
        }
    }
    if (!is.null(E)) {
        if (is.null(x)) {
            x = y
            if (!is.null(inputs$y_names)) 
                inputs$x_names = inputs$y_names
        }
        x = makelags(y = x, pop = pop, E = E, tau = tau, yname = inputs$x_names)
        if (!is.null(inputs$x_names)) {
            inputs$x_names2 = colnames(x)
            if (any(grepl("_1", inputs$x_names) | grepl("_2", 
                inputs$x_names))) {
                message("It looks like x might already contain lags, are you sure you want to be using E and tau?")
            }
        }
        inputs$E = E
        inputs$tau = tau
    }
    x = as.matrix(x)
    d = ncol(x)
    if (!is.null(augdata)) {
        xaug = as.matrix(augdata[, inputs$x_names])
        yaug = augdata[, inputs$y_names]
        timeaug = augdata[, inputs$time_names]
        if (is.null(inputs$pop_names)) {
            popaug = rep(1, nrow(augdata))
        }
        else {
            popaug = augdata[, inputs$pop_names]
        }
        yd = c(y, yaug)
        xd = rbind(x, xaug)
        timed = c(time, timeaug)
        popd = c(pop, popaug)
        primary = c(rep(T, length(y)), rep(F, length(yaug)))
    }
    else {
        yd = y
        xd = x
        timed = time
        popd = pop
        primary = rep(T, length(y))
    }
    if (scaling == "none") {
        ymeans = mean(yd, na.rm = T)
        ysds = sd(yd, na.rm = T)
        yds = yd
        xmeans = apply(xd, 2, mean, na.rm = T)
        xsds = apply(xd, 2, sd, na.rm = T)
        xds = xd
        if (ymeans > 0.1 | ymeans < (-0.1) | ysds > 1.1 | ysds < 
            (-0.9)) {
            warning("y is not scaled properly, model may be unreliable ", 
                "mean=", round(ymeans, 3), " sd=", round(ysds, 
                  3), call. = F, immediate. = T)
        }
        if (any(xmeans > 0.1) | any(xmeans < (-0.1)) | any(xsds > 
            1.1) | any(xsds < (-0.9))) {
            warning("one or more x is not scaled properly, model may be unreliable. ", 
                "means=", paste(round(xmeans, 3), collapse = " "), 
                " sds=", paste(round(xsds, 3), collapse = " "), 
                call. = F, immediate. = T)
        }
    }
    if (scaling == "global") {
        ymeans = mean(yd, na.rm = T)
        ysds = sd(yd, na.rm = T)
        yds = scale(yd)
        xmeans = apply(xd, 2, mean, na.rm = T)
        xsds = apply(xd, 2, sd, na.rm = T)
        xds = apply(xd, 2, scale)
    }
    if (scaling == "local") {
        ymeans = tapply(yd, pop, mean, na.rm = T)
        ysds = tapply(yd, pop, sd, na.rm = T)
        xlist = split(as.data.frame(xd), pop)
        xmeans = lapply(xlist, function(x) apply(x, 2, mean, 
            na.rm = T))
        xsds = lapply(xlist, function(x) apply(x, 2, sd, na.rm = T))
        up = unique(pop)
        xds = xd
        yds = yd
        for (i in 1:length(up)) {
            locmean = ymeans[as.character(up[i]) == names(ymeans)]
            locsd = ysds[as.character(up[i]) == names(ysds)]
            yds[pop == up[i]] = (yds[pop == up[i]] - locmean)/locsd
            for (j in 1:d) {
                locmean = xmeans[[which(as.character(up[i]) == 
                  names(xmeans))]][j]
                locsd = xsds[[which(as.character(up[i]) == names(xsds))]][j]
                xds[pop == up[i], j] = (xd[pop == up[i], j] - 
                  locmean)/locsd
            }
        }
    }
    up = unique(pop)
    np = length(up)
    if (np == 1) {
        stop("There must be more than 1 population to get a pairwise matrix.")
    }
    rhomat = matrix(0, nrow = np, ncol = np, dimnames = list(up, 
        up))
    for (i in 1:(np - 1)) {
        for (j in (i + 1):np) {
            popind = which(popd == up[i] | popd == up[j])
            ydsij = yds[popind]
            xdsij = xds[popind, ]
            popdij = popd[popind]
            timedij = timed[popind]
            fittemp = suppressWarnings(fitGP(y = ydsij, x = xdsij, 
                pop = popdij, time = timedij, scaling = "none", 
                initpars = initpars, modeprior = modeprior, fixedpars = fixedpars))
            rhomat[i, j] = fittemp$pars["rho"]
        }
    }
    rhomatrix = rhomat + t(rhomat) + diag(np)
    return(rhomatrix)
}
<bytecode: 0x3168ee70>
<environment: namespace:GPEDM>


#### GPnlike_fish ####
GPnlike_fish <- function (b, bmax = NULL, data, y, m, h, z, pop, time, scaling, 
    initpars, modeprior, fixedpars, rhofixed, rhomatrix, augdata, 
    linprior, xname = "escapement", ytrans) 
{
    if (!is.null(bmax)) {
        if (any(b < 0) | any(b > bmax)) {
            return(Inf)
        }
    }
    md = data[, m, drop = F]
    hd = data[, h, drop = F]
    if (length(b) > 1) {
        popvec = data[, pop]
        bvec = b[match(popvec, names(b))]
    }
    else {
        bvec = rep(b, nrow(md))
    }
    x2 = md - bvec * hd
    x = paste(xname, 1:ncol(x2), sep = "_")
    colnames(x2) = x
    data = cbind(data, x2)
    yt = paste(y, "trans", sep = "_")
    yd = ytransfun(y = data[, y], m1 = md[, 1], e1 = x2[, 1], 
        ytrans = ytrans)
    yd = data.frame(yd)
    colnames(yd) = yt
    data = cbind(data, yd)
    if (!is.null(augdata)) {
        maug = augdata[, m, drop = F]
        haug = augdata[, h, drop = F]
        if (length(b) > 1) {
            popvecaug = augdata[, pop]
            bvecaug = b[match(popvecaug, names(b))]
        }
        else {
            bvecaug = rep(b, nrow(maug))
        }
        x2aug = maug - bvecaug * haug
        colnames(x2aug) = x
        augdata = cbind(augdata, x2aug)
        ydaug = ytransfun(y = augdata[, y], m1 = maug[, 1], e1 = x2aug[, 
            1], ytrans = ytrans)
        ydaug = data.frame(ydaug)
        colnames(ydaug) = yt
        augdata = cbind(augdata, ydaug)
    }
    if (!is.null(z)) {
        x = c(x, z)
    }
    mfit = fitGP(data = data, y = yt, x = x, pop = pop, time = time, 
        scaling = scaling, initpars = initpars, modeprior = modeprior, 
        fixedpars = fixedpars, rhofixed = rhofixed, rhomatrix = rhomatrix, 
        augdata = augdata, linprior = linprior)
    return(-mfit$insampfitstats["ln_post"])
}
<bytecode: 0x316ce4c8>
<environment: namespace:GPEDM>


#### innerProd ####
innerProd <- function (x, y) 
{
    .Call("_GPEDM_innerProd", PACKAGE = "GPEDM", x, y)
}
<bytecode: 0x316edec0>
<environment: namespace:GPEDM>


#### is.wholenumber ####
is.wholenumber <- function (x, tol = .Machine$double.eps^0.5) 
{
    abs(x - round(x)) < tol
}
<bytecode: 0x316ef590>
<environment: namespace:GPEDM>


#### logit ####
logit <- function (x, min = 0, max = 1) 
{
    log((x - min)/(max - x))
}
<bytecode: 0x316f4ea0>
<environment: namespace:GPEDM>


#### makelags ####
makelags <- function (data = NULL, y, pop = NULL, E, tau, yname = NULL, forecast = FALSE, 
    vtimestep = FALSE, x = NULL, time = NULL, augment = FALSE, 
    Tdiff_max = NULL, Tdiff_fore = NULL, nreps = NULL, append = FALSE) 
{
    if (!is.wholenumber(E) | !is.wholenumber(tau) | E < 0 | tau < 
        0) {
        stop("E and tau must be positive integers")
    }
    if ((vtimestep == FALSE | forecast == TRUE) & augment == 
        TRUE) {
        stop("augment==TRUE can only be used with vtimestep==TRUE and forecast==FALSE")
    }
    if (!is.null(Tdiff_fore) & !(vtimestep == TRUE & (forecast == 
        TRUE | augment == TRUE))) {
        message("Tdiff_fore not used; only used with vtimestep==TRUE and either forecast==TRUE or augment==TRUE")
    }
    if (!is.null(Tdiff_max) & vtimestep == FALSE) {
        message("Tdiff_max not used; only used with vtimestep==TRUE")
    }
    if (!is.null(nreps) & (vtimestep == FALSE | augment == FALSE)) {
        message("nreps not used; only used with vtimestep==TRUE and augment==TRUE")
    }
    if (tau > 1 & vtimestep == TRUE) {
        message("Using tau>1 with vtimestep method is not recommended")
    }
    pname = "pop"
    tname = "time"
    xname = "xvar"
    if (!is.null(data)) {
        data = as.data.frame(data)
        yname = y
        y = data[, y]
        if (!is.null(x)) {
            xname = x
            x = data[, x]
        }
        if (!is.null(pop)) {
            pname = pop
            pop = data[, pop]
        }
        if (!is.null(time)) {
            tname = time
            time = data[, time]
        }
    }
    if (is.null(yname)) 
        yname = "var"
    y = as.matrix(y)
    num_vars = ncol(y)
    if (is.null(colnames(y))) {
        if (num_vars == length(yname)) {
            colnames(y) = yname
        }
        else {
            colnames(y) = paste0(yname, seq_len(num_vars))
        }
    }
    if (!is.null(x)) {
        x = as.matrix(x)
        if (is.null(colnames(x))) {
            if (num_vars == length(xname)) {
                colnames(x) = xname
            }
            else {
                colnames(x) = paste0(xname, seq_len(ncol(x)))
            }
        }
    }
    if (is.null(pop)) {
        pop = rep(1, nrow(y))
    }
    up = unique(pop)
    if (vtimestep == FALSE) {
        if (!is.null(x)) {
            y = cbind(y, x)
        }
        output = makelags_subrt(y = y, pop = pop, E = E, tau = tau, 
            forecast = forecast)
        if (forecast == FALSE) {
            if (append == TRUE & !is.null(data)) {
                output = cbind.data.frame(data, output)
            }
            return(output)
        }
        if (forecast == TRUE) {
            if (length(up) > 1) {
                popfore = rep(up, each = tau)
                output = cbind.data.frame(popfore, output)
                colnames(output)[1] = pname
            }
            if (!is.null(time)) {
                timefore = numeric(length(up) * tau)
                popfore = rep(up, each = tau)
                for (k in 1:length(up)) {
                  ind = which(pop == up[k])
                  indfore = which(popfore == up[k])
                  timei = time[ind]
                  Tdiff = diff(timei[(length(timei) - 1):length(timei)])
                  timefore[indfore] = timei[length(timei)] + 
                    (1:tau) * Tdiff
                }
                output = cbind.data.frame(timefore, output)
                colnames(output)[1] = tname
            }
            return(output)
        }
    }
    if (vtimestep == TRUE) {
        if (is.null(time)) {
            time = y * 0
            for (i in 1:length(up)) {
                time[pop == up[i]] = 1:length(time[pop == up[i]])
            }
        }
        if (!is.null(x)) {
            xy = cbind(y, x)
        }
        else {
            xy = y
        }
        outlist = NULL
        for (k in 1:length(up)) {
            pind = which(pop == up[k])
            xyi = xy[pind, , drop = F]
            timei = time[pind]
            nm = which(apply(xyi, 1, function(x) all(!is.na(x))))
            out = matrix(nrow = nrow(xyi), ncol = E * (ncol(xyi) + 
                1))
            vnames = colnames(xyi)
            colnames(out) = paste(rep(c(vnames, "Tdiff"), each = E), 
                rep(seq(tau, tau * E, tau), times = length(vnames) + 
                  1), sep = "_")
            if (forecast) {
                if (is.null(Tdiff_fore)) {
                  Tdiff_fore = min(diff(timei)) * tau
                }
                foredf = matrix(NA, ncol = ncol(xyi), nrow = length(Tdiff_fore))
                colnames(foredf) = colnames(xyi)
                xy2 = rbind(xyi, foredf)
                time2 = c(timei, timei[length(timei)] + Tdiff_fore)
                out = matrix(nrow = nrow(xyi) + length(Tdiff_fore), 
                  ncol = E * (ncol(xyi) + 1))
                colnames(out) = paste(rep(c(vnames, "Tdiff"), 
                  each = E), rep(seq(tau, tau * E, tau), times = length(vnames) + 
                  1), sep = "_")
                indfore = (nrow(xyi) + 1):nrow(xy2)
            }
            else {
                xy2 = xyi
                time2 = timei
            }
            for (i in 1:nrow(xy2)) {
                if (i > E * tau) {
                  index = numeric(length = E + 1)
                  index[1] = i
                  indiff = i - nm
                  for (Ei in 1:E) {
                    newind = nm[which.min(indiff[indiff >= tau * 
                      Ei & nm <= index[Ei] - tau])]
                    if (length(newind) != 0) {
                      index[Ei + 1] = newind
                      out[i, 0:(length(vnames) - 1) * E + Ei] = as.numeric(xyi[index[Ei + 
                        1], ])
                      out[i, length(vnames) * E + Ei] = time2[index[Ei]] - 
                        time2[index[Ei + 1]]
                    }
                  }
                }
            }
            if (forecast) {
                out = cbind.data.frame(time2[indfore], out[indfore, 
                  , drop = F])
                colnames(out)[1] = tname
                if (length(up) > 1) {
                  out = cbind.data.frame(rep(up[k], nrow(out)), 
                    out)
                  colnames(out)[1] = pname
                }
                outlist[[k]] = out
            }
            else {
                outlist[[k]] = out
            }
        }
        output = do.call(rbind, outlist)
        if (augment == FALSE & append == TRUE & !is.null(data)) {
            output = cbind.data.frame(data, output)
        }
        if (!is.null(Tdiff_max)) {
            cols = grep("Tdiff_", colnames(output))
            output[, cols][output[, cols] > Tdiff_max] <- NA
        }
        if (augment == TRUE) {
            if (is.null(nreps)) {
                message("defaulting to nreps=10")
                nreps = 10
            }
            cols = grep("Tdiff_", colnames(output))
            auglist = NULL
            for (k in 1:length(up)) {
                pind = which(pop == up[k])
                xyi = xy[pind, , drop = F]
                yi = y[pind, ]
                timei = time[pind]
                outputi = outlist[[k]]
                outputi2 = cbind(outputi, yi)
                outbase = as.data.frame(na.omit(outputi2))[, 
                  cols, drop = F]
                outbase$combo = apply(outbase, 1, paste, collapse = "")
                outtab = as.data.frame(table(combo = outbase$combo))
                outtab = unique(merge(outbase, outtab))
                outtab = outtab[, -1]
                if (!is.null(Tdiff_fore)) {
                  Tgrid = expand.grid(c(rep(list(Tdiff_fore), 
                    E)))
                  colnames(Tgrid) = paste0("Tdiff_", 1:E)
                  outtab = merge(outtab, Tgrid, all = T)
                  outtab$Freq[is.na(outtab$Freq)] = 0
                }
                if (!is.null(Tdiff_max)) {
                  outtab = outtab[apply(outtab[, 1:E], 1, max) <= 
                    Tdiff_max, ]
                }
                augmat = outputi[0, ]
                newrow0 = outputi[1, ]
                timeaug = numeric()
                yaug = numeric()
                outtab$Freq_new = NA
                rownames(outtab) = NULL
                for (i in 1:nrow(outtab)) {
                  freq = outtab$Freq[i]
                  remainingrows = nrow(xyi)
                  randind = sample(1:nrow(xyi), nrow(xyi), replace = F)
                  randindi = 1
                  while (freq < nreps & randindi <= nrow(xyi)) {
                    if (randind[randindi] > E * tau & !is.na(yi[randind[randindi]])) {
                      index = numeric(length = E + 1)
                      index[1] = randind[randindi]
                      tdiffs = as.numeric(outtab[i, grep("Tdiff_", 
                        colnames(outtab))])
                      newrow = newrow0
                      for (Ei in 1:E) {
                        index[Ei + 1] = index[Ei] - tdiffs[Ei]
                        if (index[Ei + 1] > 0) {
                          newrow[0:(length(vnames) - 1) * E + 
                            Ei] = as.numeric(xyi[index[Ei + 1], 
                            ])
                          newrow[length(vnames) * E + Ei] = time[index[Ei]] - 
                            time[index[Ei + 1]]
                        }
                      }
                      if (all(!is.na(newrow)) & !duplicated(rbind(outputi, 
                        newrow))[nrow(outputi) + 1]) {
                        augmat = rbind(augmat, newrow)
                        timeaug = c(timeaug, time[randind[randindi]])
                        yaug = c(yaug, yi[randind[randindi]])
                        freq = freq + 1
                      }
                    }
                    randindi = randindi + 1
                  }
                  outtab$Freq_new[i] = freq
                }
                rownames(augmat) = NULL
                out = cbind.data.frame(timeaug, yaug, augmat)
                colnames(out)[1:2] = c(tname, yname[1])
                if (length(up) > 1) {
                  out = cbind.data.frame(rep(up[k], nrow(out)), 
                    out)
                  colnames(out)[1] = pname
                }
                auglist[[k]] = out
                cat("Population ", up[k], "\n")
                if (nrow(out) == 0) {
                  message("Additional combinations not available. Check nreps or Tdiff_fore for more options.")
                }
                print(outtab)
            }
            output = do.call(rbind, auglist)
        }
        return(output)
    }
}
<bytecode: 0x31728958>
<environment: namespace:GPEDM>


#### makelags_iter ####
makelags_iter <- function (data, nfore, y, pop = NULL, E, tau, time, ...) 
{
    skipcols = ifelse(is.null(pop), 1, 2)
    foremat1 = makelags(data, y = y, pop = pop, time = time, 
        tau = tau, E = E, forecast = T, ...)
    if (is.null(pop)) {
        foremat2 = data.frame(time = (foremat1[1, time] + 1):(foremat1[1, 
            time] + nfore - 1))
        colnames(foremat2) = time
    }
    else {
        foremat2 = expand.grid(time = (foremat1[1, time] + 1):(foremat1[1, 
            time] + nfore - 1), pop = unique(data[, pop]))
        colnames(foremat2) = c(time, pop)
    }
    forecols = colnames(foremat1)[-(1:skipcols)]
    foremat3 = matrix(NA, nrow = nrow(foremat2), ncol = ncol(foremat1) - 
        skipcols, dimnames = list(NULL, forecols))
    foremat = rbind(foremat1, cbind(foremat2, foremat3))
    return(foremat)
}
<bytecode: 0x317a5db8>
<environment: namespace:GPEDM>


#### makelags_subrt ####
makelags_subrt <- function (y, pop, E, tau, forecast) 
{
    up = unique(pop)
    num_vars = ncol(y)
    output = matrix(NA, nrow = nrow(y), ncol = num_vars * E)
    outputfore = matrix(NA, nrow = length(up) * tau, ncol = num_vars * 
        E)
    popfore = rep(up, each = tau)
    col_names = character(num_vars * E)
    for (k in 1:length(up)) {
        col_index = 1
        for (j in 1:num_vars) {
            ind = which(pop == up[k])
            indfore = which(popfore == up[k])
            ts = y[ind, j]
            if (length(ts) <= E * tau) 
                stop("time series length <= E*tau for pop ", 
                  up[k])
            for (i in 1:E) {
                tslag = c(rep_len(NA, tau * i), ts[1:(length(ts) - 
                  tau * i)])
                output[ind, col_index] = tslag
                tsfore = ts[(length(ts) - tau * i + 1):(length(ts) - 
                  tau * (i - 1))]
                outputfore[indfore, col_index] = tsfore
                col_names[col_index] = paste0(colnames(y)[j], 
                  "_", i * tau)
                col_index = col_index + 1
            }
        }
    }
    colnames(output) = col_names
    colnames(outputfore) = col_names
    if (forecast == FALSE) 
        return(output)
    else return(outputfore)
}
<bytecode: 0x317c5b80>
<environment: namespace:GPEDM>


#### matrix2vector ####
matrix2vector <- function (m) 
{
    .Call("_GPEDM_matrix2vector", PACKAGE = "GPEDM", m)
}
<bytecode: 0x317d4a50>
<environment: namespace:GPEDM>


#### msy_wrapper ####
msy_wrapper <- function (model, newdata, hratevec, tsave) 
{
    up = unique(model$inputs$pop)
    catch_1 = model$inputs$h_names[1]
    msyfore = list()
    catchsavepop = expand.grid(pop = up, time = 1:tsave, hrate = hratevec)
    catchsavepop$catch = NA
    catchsavepop$cpue = NA
    for (h in seq_along(hratevec)) {
        msyfore[[h]] = predict_iter(model, newdata = newdata, 
            hrate = hratevec[h])$outsampresults
        msyfore[[h]] = cbind.data.frame(hrate = hratevec[h], 
            msyfore[[h]])
        for (p in 1:length(up)) {
            msyforei = subset(msyfore[[h]], pop == up[p])
            nfore = nrow(msyforei)
            if (tsave > nfore) {
                tsavei = nfore
            }
            else {
                tsavei = tsave
            }
            catchsavepop$catch[catchsavepop$time %in% 1:tsavei & 
                catchsavepop$hrate == hratevec[h] & catchsavepop$pop == 
                up[p]] = msyforei[(nfore - tsavei + 1):nfore, 
                catch_1]
            catchsavepop$cpue[catchsavepop$time %in% 1:tsavei & 
                catchsavepop$hrate == hratevec[h] & catchsavepop$pop == 
                up[p]] = msyforei$predmean[(nfore - tsavei + 
                1):nfore]
        }
    }
    catchforeall = do.call(rbind, msyfore)
    catchsavetotal = aggregate(cbind(catch, cpue) ~ hrate * time, 
        data = catchsavepop, sum)
    catchsavepopmean = aggregate(cbind(catch, cpue) ~ hrate * 
        pop, data = catchsavepop, mean)
    catchsavetotalmean = aggregate(cbind(catch, cpue) ~ hrate, 
        data = catchsavetotal, mean)
    fmsy = catchsavetotalmean$hrate[which.max(catchsavetotalmean$catch)]
    Bmsy = catchsavetotalmean$cpue[which.max(catchsavetotalmean$catch)]
    return(list(catchforeall = catchforeall, catchsavepop = catchsavepop, 
        catchsavetotal = catchsavetotal, catchsavepopmean = catchsavepopmean, 
        catchsavetotalmean = catchsavetotalmean, fmsy = fmsy, 
        Bmsy = Bmsy))
}
<bytecode: 0x317d7748>
<environment: namespace:GPEDM>


#### multMat ####
multMat <- function (m1, m2) 
{
    .Call("_GPEDM_multMat", PACKAGE = "GPEDM", m1, m2)
}
<bytecode: 0x317eec98>
<environment: namespace:GPEDM>


#### plot.GPpred ####
plot.GPpred <- function (x, plotinsamp = FALSE, ytrans = FALSE, ...) 
{
    if (is.null(x$outsampresults) | plotinsamp) {
        dplot = x$insampresults
        message("Plotting in sample results.")
    }
    else {
        dplot = x$outsampresults
        message("Plotting out of sample results.")
    }
    up = unique(dplot$pop)
    np = length(up)
    if (!is.null(x$inputs$y_names)) {
        yl = x$inputs$y_names
        if (!is.null(x$b)) {
            ylt = yl
            yl = sub("_trans", "", yl)
        }
    }
    else {
        yl = "y"
    }
    old.par <- par(no.readonly = TRUE)
    on.exit(par(old.par), add = T, after = F)
    cols = ifelse(is.null(dplot$obs), 1, 2)
    par(mfrow = c(min(4, np), cols), mar = c(5, 4, 2, 2))
    if (ytrans == FALSE) {
        for (i in 1:np) {
            dploti = subset(dplot, pop == up[i])
            plot(dploti$timestep, dploti$predmean, type = "o", 
                ylab = yl, xlab = "Time", ylim = range(dploti$predmean + 
                  dploti$predfsd, dploti$predmean - dploti$predfsd, 
                  dploti$obs, na.rm = T), main = up[i])
            lines(dploti$timestep, dploti$predmean + dploti$predfsd, 
                lty = 2)
            lines(dploti$timestep, dploti$predmean - dploti$predfsd, 
                lty = 2)
            points(dploti$timestep, dploti$obs, col = "blue")
            legend(x = "bottomright", legend = c("obs", "pred"), 
                col = c("blue", "black"), pch = 1)
            if (!is.null(dploti$obs)) {
                plot(dploti$obs, dploti$predmean, ylab = paste(yl, 
                  "pred"), xlab = paste(yl, "obs"), main = up[i])
                abline(a = 0, b = 1)
            }
        }
    }
    else {
        for (i in 1:np) {
            dploti = subset(dplot, pop == up[i])
            plot(dploti$timestep, dploti$predmean_trans, type = "o", 
                ylab = ylt, xlab = "Time", ylim = range(dploti$predmean_trans + 
                  dploti$predfsd_trans, dploti$predmean_trans - 
                  dploti$predfsd_trans, dploti$obs_trans, na.rm = T), 
                main = up[i])
            lines(dploti$timestep, dploti$predmean_trans + dploti$predfsd_trans, 
                lty = 2)
            lines(dploti$timestep, dploti$predmean_trans - dploti$predfsd_trans, 
                lty = 2)
            points(dploti$timestep, dploti$obs_trans, col = "blue")
            legend(x = "bottomright", legend = c("obs", "pred"), 
                col = c("blue", "black"), pch = 1)
            if (!is.null(dploti$obs_trans)) {
                plot(dploti$obs_trans, dploti$predmean_trans, 
                  ylab = paste(ylt, "pred"), xlab = paste(ylt, 
                    "obs"), main = up[i])
                abline(a = 0, b = 1)
            }
        }
    }
}
<bytecode: 0x317f4f28>
<environment: namespace:GPEDM>


#### posdef ####
posdef <- function (mat) 
{
    matpd = Matrix::nearPD(mat, corr = T)
    matpd = as.matrix(matpd$mat)
    return(matpd)
}
<bytecode: 0x318144e0>
<environment: namespace:GPEDM>


#### predict_iter ####
predict_iter <- function (object, newdata, xlags = NULL, hrate = NULL) 
{
    if (is.null(object$inputs$time_names)) {
        stop("Model must include `data` and `time` to use this function.")
    }
    if (!is.null(object$inputs$E)) {
        stop("Lags must be pre-generated to use this function. See option A1 in help(fitGP).")
    }
    if (!(object$inputs$time_names %in% colnames(newdata))) {
        stop("Time column '", object$inputs$time_names, "' not found in newdata.")
    }
    if (!is.null(object$b)) {
        if (!is.null(xlags)) {
            message("xlags in fisheries models set to `m` variables, input ignored")
        }
        xlags = object$inputs$m_names
        hlags = object$inputs$h_names
        b = object$b
    }
    if (is.null(xlags)) {
        xlags = object$inputs$x_names
    }
    timename = object$inputs$time_names
    newtimes = unique(newdata[, timename])
    if (is.null(object$inputs$pop_names)) {
        popname = "pop"
        newdata$pop = 1
    }
    else {
        popname = object$inputs$pop_names
    }
    if (!(popname %in% colnames(newdata))) {
        stop("Pop column '", popname, "' not found in newdata.")
    }
    up = unique(newdata[, popname])
    if (!is.null(object$b)) {
        if (length(b) == 1 & length(up) > 1) {
            b = rep(b, length(up))
        }
        if (is.null(names(b))) {
            names(b) = up
        }
    }
    pred = list()
    for (i in seq_along(newtimes)) {
        newdatai = newdata[newdata[, timename] == newtimes[i], 
            ]
        pred[[i]] = predict(object, newdata = newdatai)$outsampresults
        preds = pred[[i]]
        if (any(is.infinite(preds$predmean))) {
            message("predicted y is infinite at timestep ", newtimes[i], 
                ". Stopping iteration.")
            break
        }
        if (i + 1 <= length(newtimes)) {
            for (j in 1:length(up)) {
                predsj = preds[preds$pop == up[j], ]$predmean
                newdata[newdata[, timename] == newtimes[i + 1] & 
                  newdata[, popname] == up[j], xlags[1]] = predsj
                if (length(xlags) > 1) {
                  newdata[newdata[, timename] == newtimes[i + 
                    1] & newdata[, popname] == up[j], xlags[2:length(xlags)]] = newdata[newdata[, 
                    timename] == newtimes[i] & newdata[, popname] == 
                    up[j], xlags[1:(length(xlags) - 1)]]
                }
                if (!is.null(hrate)) {
                  newdata[newdata[, timename] == newtimes[i + 
                    1] & newdata[, popname] == up[j], hlags[1]] = predsj/b[as.character(up[j])] * 
                    hrate
                  if (length(xlags) > 1) {
                    newdata[newdata[, timename] == newtimes[i + 
                      1] & newdata[, popname] == up[j], hlags[2:length(xlags)]] = newdata[newdata[, 
                      timename] == newtimes[i] & newdata[, popname] == 
                      up[j], hlags[1:(length(xlags) - 1)]]
                  }
                }
            }
        }
    }
    outsampresults = do.call(rbind, pred)
    out = list(outsampresults = outsampresults)
    if (!is.null(hrate)) {
        out$outsampresults = merge(out$outsampresults, newdata[, 
            c(timename, popname, hlags), drop = F], by.x = c("timestep", 
            "pop"), by.y = c(timename, popname))
    }
    if (object$inputs$y_names %in% colnames(newdata)) {
        if (!all(is.na(outsampresults$obs))) {
            out$outsampfitstats = c(R2 = getR2(outsampresults$obs, 
                outsampresults$predmean), rmse = sqrt(mean((outsampresults$obs - 
                outsampresults$predmean)^2, na.rm = T)))
            if (length(unique(outsampresults$pop)) > 1) {
                out$outsampfitstatspop = getR2pop(outsampresults$obs, 
                  outsampresults$predmean, outsampresults$pop)
                R2centered = getR2pop(outsampresults$obs, outsampresults$predmean, 
                  outsampresults$pop, type = "centered")
                R2scaled = getR2pop(outsampresults$obs, outsampresults$predmean, 
                  outsampresults$pop, type = "scaled")
                out$outsampfitstats = c(out$outsampfitstats, 
                  R2centered = R2centered, R2scaled = R2scaled)
            }
        }
    }
    class(out) = "GPpred"
    return(out)
}
<bytecode: 0x3181a410>
<environment: namespace:GPEDM>


#### predict_seq ####
predict_seq <- function (object, newdata, restart.initpars = F, refit.b = F) 
{
    if (is.null(object$inputs$time_names)) {
        stop("Model must include `data` and `time` to use this function.")
    }
    if (!is.null(object$inputs$E)) {
        stop("Lags must be pre-generated to use this function. See option A1 in help(fitGP).")
    }
    if (refit.b) {
        bfixed = NULL
    }
    else {
        bfixed = object$b
    }
    timename = object$inputs$time_names
    newtimes = unique(newdata[, timename])
    ogdata = eval(object$call$data)
    modelupdated = object
    dataupdated = ogdata
    pred = list()
    for (i in seq_along(newtimes)) {
        newdatai = newdata[newdata[, timename] == newtimes[i], 
            ]
        pred[[i]] = predict(modelupdated, newdata = newdatai)$outsampresults
        dataupdated = rbind(dataupdated, newdatai)
        if (restart.initpars) {
            inits = NULL
        }
        else {
            inits = modelupdated$pars
        }
        if (!is.null(bfixed)) {
            modelupdated = update(modelupdated, data = dataupdated, 
                initpars = inits, bfixed = bfixed)
        }
        else {
            modelupdated = update(modelupdated, data = dataupdated, 
                initpars = inits)
        }
    }
    outsampresults = do.call(rbind, pred)
    modelupdated$outsampresults = outsampresults
    modelupdated$outsampfitstats = c(R2 = getR2(outsampresults$obs, 
        outsampresults$predmean), rmse = sqrt(mean((outsampresults$obs - 
        outsampresults$predmean)^2, na.rm = T)))
    if (length(unique(outsampresults$pop)) > 1) {
        modelupdated$outsampfitstatspop = getR2pop(outsampresults$obs, 
            outsampresults$predmean, outsampresults$pop)
        R2centered = getR2pop(outsampresults$obs, outsampresults$predmean, 
            outsampresults$pop, type = "centered")
        R2scaled = getR2pop(outsampresults$obs, outsampresults$predmean, 
            outsampresults$pop, type = "scaled")
        modelupdated$outsampfitstats = c(modelupdated$outsampfitstats, 
            R2centered = R2centered, R2scaled = R2scaled)
    }
    return(modelupdated)
}
<bytecode: 0x31849290>
<environment: namespace:GPEDM>


#### predict.GP ####
predict.GP <- function (object, predictmethod = c("loo", "lto", "sequential"), 
    newdata = NULL, xnew = NULL, popnew = NULL, timenew = NULL, 
    ynew = NULL, returnGPgrad = FALSE, exclradius = 0, ...) 
{
    iKVs = object$covm$iKVs
    phi = object$pars[grepl("phi", names(object$pars))]
    sigma2 = object$pars[names(object$pars) == "sigma2"]
    rho = object$pars[names(object$pars) == "rho"]
    ve = object$pars[names(object$pars) == "ve"]
    X = object$inputs$X
    Y = object$inputs$Y
    y = object$inputs$y
    rhomatrix = object$inputs$rhomatrix
    Pop = object$inputs$Pop
    scaling = object$scaling$scaling
    ymeans = object$scaling$ymeans
    ysds = object$scaling$ysds
    xmeans = object$scaling$xmeans
    xsds = object$scaling$xsds
    if (!is.null(newdata) | !is.null(xnew)) {
        if (!is.null(object$b)) {
            if (!is.null(xnew)) {
                stop("Must use `newdata` for fisheries models")
            }
            if (!all(object$inputs$x_names %in% colnames(newdata))) {
                b = object$b
                md = newdata[, object$inputs$m_names, drop = F]
                hd = newdata[, object$inputs$h_names, drop = F]
                if (length(b) > 1) {
                  popvec = newdata[, object$inputs$pop_names]
                  bvec = b[match(popvec, names(b))]
                }
                else {
                  bvec = rep(b, nrow(md))
                }
                x2 = md - bvec * hd
                colnames(x2) = object$inputs$x_names[1:ncol(x2)]
                newdata = cbind(newdata, x2)
            }
            else {
                x2 = newdata[object$inputs$x_names[1:length(object$inputs$m_names)]]
            }
            if (object$inputs$y0_names %in% colnames(newdata)) {
                y0 = object$inputs$y0
                yd = ytransfun(y = newdata[, y0], m1 = md[, 1], 
                  e1 = x2[, 1], ytrans = object$inputs$ytrans)
                yd = data.frame(yd)
                colnames(yd) = y
                newdata = cbind(newdata, yd)
            }
        }
        if (!is.null(newdata)) {
            if (is.character(object$inputs$y_names)) {
                if (object$inputs$y_names %in% colnames(newdata)) {
                  ynew = newdata[, object$inputs$y_names]
                }
            }
            if (is.numeric(object$inputs$y_names)) {
                if (object$inputs$y_names <= ncol(newdata)) {
                  ynew = newdata[, object$inputs$y_names]
                }
            }
            if (!is.null(object$inputs$x_names)) {
                xnew = newdata[, object$inputs$x_names]
            }
            if (!is.null(object$inputs$pop_names)) {
                popnew = newdata[, object$inputs$pop_names]
            }
            if (!is.null(object$inputs$time_names)) {
                timenew = newdata[, object$inputs$time_names]
            }
        }
        if (!is.null(ynew)) {
            Tslp = length(ynew)
        }
        else {
            Tslp = nrow(as.matrix(xnew))
        }
        if (is.null(popnew)) {
            popnew = rep(1, Tslp)
        }
        if (is.null(timenew)) {
            up = unique(popnew)
            timenew = numeric(Tslp)
            for (i in 1:length(up)) {
                timenew[popnew == up[i]] = 1:length(timenew[popnew == 
                  up[i]])
            }
        }
        if (!is.null(object$inputs$E)) {
            if (is.null(xnew)) {
                xnew = ynew
            }
            xnew = makelags(y = xnew, pop = popnew, E = object$inputs$E, 
                tau = object$inputs$tau, yname = object$inputs$x_names)
        }
        xnew = as.matrix(xnew)
        if (scaling == "none") {
            xnews = xnew
        }
        if (scaling == "global") {
            xnews = xnew
            for (j in 1:ncol(xnew)) {
                xnews[, j] = (xnew[, j] - xmeans[j])/xsds[j]
            }
        }
        if (scaling == "local") {
            up = unique(popnew)
            xnews = xnew
            for (i in 1:length(up)) {
                for (j in 1:ncol(xnew)) {
                  locmean = xmeans[[which(as.character(up[i]) == 
                    names(xmeans))]][j]
                  locsd = xsds[[which(as.character(up[i]) == 
                    names(xsds))]][j]
                  xnews[popnew == up[i], j] = (xnew[popnew == 
                    up[i], j] - locmean)/locsd
                }
            }
        }
        completerowsnew = complete.cases(xnews)
        if (all(completerowsnew == FALSE)) {
            out = list(outsampresults = data.frame(timestep = timenew, 
                pop = popnew, predmean = NA, predfsd = NA, predsd = NA))
            if (!is.null(ynew)) {
                out$outsampresults$obs = ynew
            }
            class(out) = "GPpred"
            return(out)
        }
        Xnew = as.matrix(xnews[completerowsnew, , drop = F])
        Popnew = popnew[completerowsnew]
        covmnew = getcov(phi, sigma2, rho, X, Xnew, Pop, Popnew, 
            rhomatrix)
        Cs = covmnew$Cd
        ymean = Cs %*% (iKVs %*% Y)
        yvar = numeric(length = nrow(Xnew))
        for (i in 1:nrow(Xnew)) {
            yvar[i] = sigma2 - Cs[i, ] %*% iKVs %*% Cs[i, ]
        }
        if (returnGPgrad) {
            GPgrad = Xnew * NA
            for (i in 1:nrow(Xnew)) {
                Xnewi = Xnew[i, ]
                Csi = Cs[i, , drop = FALSE]
                GPgrad[i, ] = getGPgrad(phi = phi, Xnew = Xnewi, 
                  X = X, Cdi = Csi, iKVs = iKVs, Y = Y)
            }
            gpgrad = xnew * NA
            if (!is.null(object$inputs$x_names2)) {
                xnames = object$inputs$x_names2
            }
            else {
                xnames = object$inputs$x_names
            }
            colnames(gpgrad) = paste0("d_", xnames)
            gpgrad[completerowsnew, ] = GPgrad
        }
        ypred <- ysd <- yfsd <- numeric(Tslp) * NA
        ypred[completerowsnew] = ymean
        yfsd[completerowsnew] = sqrt(yvar)
        ysd[completerowsnew] = sqrt(yvar + ve)
    }
    else {
        predictmethod = match.arg(predictmethod)
        if (!is.null(object$b)) {
            md = object$inputs$m
            x2 = object$inputs$x
            colnames(x2) = object$inputs$x_names[1:ncol(x2)]
        }
        if (predictmethod == "loo") {
            Cd = object$covm$Cd
            Sigma = object$covm$Sigma
            Time = object$inputs$Time
            Primary = object$inputs$Primary
            nd = length(which(Primary))
            ymean = numeric(length = nd)
            yvar = numeric(length = nd)
            if (returnGPgrad) {
                GPgrad = matrix(NA, nrow = nd, ncol = ncol(X))
            }
            for (i in 1:nd) {
                exclude = (i - exclradius):(i + exclradius)
                exclude = exclude[exclude %in% which(Pop == Pop[i])]
                dups = which(paste0(Pop[!Primary], Time[!Primary]) %in% 
                  paste0(Pop[exclude], Time[exclude])) + nd
                Cdi = Cd[i, -c(exclude, dups), drop = F]
                S_noi = Sigma[-c(exclude, dups), -c(exclude, 
                  dups)]
                Y_noi = Y[-c(exclude, dups)]
                S_noi = (S_noi + t(S_noi))/2
                icov_noi = getcovinv(S_noi)
                iKVs_noi = icov_noi$iKVs
                ymean[i] = Cdi %*% iKVs_noi %*% Y_noi
                yvar[i] = sigma2 - Cdi %*% iKVs_noi %*% t(Cdi)
                if (returnGPgrad) {
                  Xi = X[i, ]
                  Xnoi = X[-c(exclude, dups), ]
                  GPgrad[i, ] = getGPgrad(phi = phi, Xnew = Xi, 
                    X = Xnoi, Cdi = Cdi, iKVs = iKVs_noi, Y = Y_noi)
                }
            }
            ypred <- ysd <- yfsd <- y * NA
            ypred[object$inputs$completerows[1:length(y)]] = ymean
            yfsd[object$inputs$completerows[1:length(y)]] = sqrt(yvar)
            ysd[object$inputs$completerows[1:length(y)]] = sqrt(yvar + 
                ve)
            if (returnGPgrad) {
                if (!is.null(object$inputs$x_names2)) {
                  xnames = object$inputs$x_names2
                }
                else {
                  xnames = object$inputs$x_names
                }
                gpgrad = matrix(NA, nrow = length(y), ncol = ncol(X), 
                  dimnames = list(NULL, paste0("d_", xnames)))
                gpgrad[object$inputs$completerows[1:length(y)], 
                  ] = GPgrad
            }
            popnew = object$inputs$pop
            timenew = object$inputs$time
            ynew = y
        }
        if (predictmethod == "lto") {
            Time = object$inputs$Time
            timevals = unique(Time)
            nt = length(timevals)
            Cd = object$covm$Cd
            Sigma = object$covm$Sigma
            Primary = object$inputs$Primary
            nd = length(which(Primary))
            ymean = numeric(length = nd) * NA
            yvar = numeric(length = nd) * NA
            if (returnGPgrad) {
                GPgrad = matrix(NA, nrow = nd, ncol = ncol(X))
            }
            for (i in 1:nt) {
                exclude = (i - exclradius):(i + exclradius)
                exclude = exclude[exclude > 0 & exclude <= nt]
                ind = which(Time[Primary] == timevals[i])
                train = which(!(Time %in% timevals[exclude]))
                Cdi = Cd[ind, train, drop = F]
                S_noi = Sigma[train, train]
                Y_noi = Y[train]
                S_noi = (S_noi + t(S_noi))/2
                icov_noi = getcovinv(S_noi)
                iKVs_noi = icov_noi$iKVs
                ymean[ind] = Cdi %*% iKVs_noi %*% Y_noi
                yvar[ind] = diag(sigma2 - Cdi %*% iKVs_noi %*% 
                  t(Cdi))
                if (returnGPgrad) {
                  for (j in ind) {
                    Xi = X[j, ]
                    Xnoi = X[train, ]
                    Cdii = Cd[j, train, drop = F]
                    GPgrad[j, ] = getGPgrad(phi = phi, Xnew = Xi, 
                      X = Xnoi, Cdi = Cdii, iKVs = iKVs_noi, 
                      Y = Y_noi)
                  }
                }
            }
            ypred <- ysd <- yfsd <- y * NA
            ypred[object$inputs$completerows[1:length(y)]] = ymean
            yfsd[object$inputs$completerows[1:length(y)]] = sqrt(yvar)
            ysd[object$inputs$completerows[1:length(y)]] = sqrt(yvar + 
                ve)
            if (returnGPgrad) {
                if (!is.null(object$inputs$x_names2)) {
                  xnames = object$inputs$x_names2
                }
                else {
                  xnames = object$inputs$x_names
                }
                gpgrad = matrix(NA, nrow = length(y), ncol = ncol(X), 
                  dimnames = list(NULL, paste0("d_", xnames)))
                gpgrad[object$inputs$completerows[1:length(y)], 
                  ] = GPgrad
            }
            popnew = object$inputs$pop
            timenew = object$inputs$time
            ynew = y
        }
        if (predictmethod == "sequential") {
            Time = object$inputs$Time
            up = unique(Pop)
            np = length(up)
            timevals = unique(Time)
            nt = length(timevals)
            S = object$covm$Cd
            Primary = object$inputs$Primary
            nd = length(which(Primary))
            ymean1 = matrix(0, nrow = length(Y), ncol = nt)
            ysd1 = (sigma2 + ve) * matrix(1, nrow = length(Y), 
                ncol = nt)
            if (returnGPgrad) {
                message("returnGPgrad not available for predictmethod='sequential'")
                returnGPgrad = FALSE
            }
            for (i in 1:(nt - 1)) {
                ind = which(Time == timevals[i])
                I = diag(length(ind))
                SS = S[ind, ind] + ve * I
                SS = (SS + t(SS))/2
                icov_ss = getcovinv(SS)
                iK = icov_ss$iKVs
                k = iK %*% (Y[ind] - ymean1[ind, i])
                ymean1[, i + 1] = ymean1[, i] + S[, ind] %*% 
                  k
                S = S - S[, ind] %*% iK %*% S[ind, ]
                ysd1[, i + 1] = sqrt(abs(diag(S)) + ve)
            }
            ymean = rep(NA, length(Y))
            ysd2 = rep((sigma2 + ve), length(Y))
            for (i in 1:nt) {
                for (k in 1:np) {
                  ind = which(Time == timevals[i] & Pop == up[k])
                  if (length(ind) > 0) {
                    ymean[ind] = ymean1[ind, i]
                    ysd2[ind] = ysd1[ind, i]
                  }
                }
            }
            ypred <- ysd <- yfsd <- y * NA
            ypred[object$inputs$completerows[1:length(y)]] = ymean[Primary]
            yfsd[object$inputs$completerows[1:length(y)]] = sqrt(ysd2^2 - 
                ve)[Primary]
            ysd[object$inputs$completerows[1:length(y)]] = ysd2[Primary]
            popnew = object$inputs$pop
            timenew = object$inputs$time
            ynew = y
        }
    }
    if (scaling == "global") {
        ypred = ypred * ysds + ymeans
        yfsd = sqrt(yfsd^2 * ysds^2)
        ysd = sqrt(ysd^2 * ysds^2)
        if (returnGPgrad) {
            for (i in 1:ncol(gpgrad)) {
                gpgrad[, i] = gpgrad[, i] * ysds/object$scaling$xsds[i]
            }
        }
    }
    if (scaling == "local") {
        up = unique(popnew)
        for (i in 1:length(up)) {
            locmean = ymeans[as.character(up[i]) == names(ymeans)]
            locsd = ysds[as.character(up[i]) == names(ysds)]
            ypred[popnew == up[i]] = ypred[popnew == up[i]] * 
                locsd + locmean
            yfsd[popnew == up[i]] = sqrt(yfsd[popnew == up[i]]^2 * 
                locsd^2)
            ysd[popnew == up[i]] = sqrt(ysd[popnew == up[i]]^2 * 
                locsd^2)
            if (returnGPgrad) {
                for (j in 1:ncol(gpgrad)) {
                  xsds = object$scaling$xsds
                  locsdx = xsds[[which(as.character(up[i]) == 
                    names(xsds))]][j]
                  gpgrad[popnew == up[i], j] = gpgrad[popnew == 
                    up[i], j] * locsd/locsdx
                }
            }
        }
    }
    if (!is.null(object$linprior)) {
        if (!is.null(xnew)) {
            regdf_new = data.frame(x = xnew[, 1], pop = popnew)
            ynewlinprior = predict(object$linprior$linprior_reg, 
                newdata = regdf_new)
            ypred = ypred + ynewlinprior
        }
        else {
            ynewlinprior = object$linprior$ylinprior
            ypred = ypred + ynewlinprior
        }
    }
    out = list(outsampresults = data.frame(timestep = timenew, 
        pop = popnew, predmean = ypred, predfsd = yfsd, predsd = ysd))
    if (!is.null(ynew)) {
        out$outsampresults = cbind(out$outsampresults, obs = ynew)
    }
    if (!is.null(object$b)) {
        ypred_trans = ypred
        ypred = ytransfuninv(y = ypred_trans, m1 = md[, 1], e1 = x2[, 
            1], ytrans = object$inputs$ytrans)
        if (!is.null(ynew)) {
            ynew_trans = ynew
            ynew = ytransfuninv(y = ynew_trans, m1 = md[, 1], 
                e1 = x2[, 1], ytrans = object$inputs$ytrans)
        }
        out = list(outsampresults = data.frame(timestep = timenew, 
            pop = popnew, predmean_trans = ypred_trans, predfsd_trans = yfsd, 
            predsd_trans = ysd))
        if (!is.null(ynew)) {
            out$outsampresults = cbind(out$outsampresults, obs_trans = ynew_trans)
        }
        out$outsampresults = cbind(out$outsampresults, predmean = ypred)
        if (!is.null(ynew)) {
            out$outsampresults = cbind(out$outsampresults, obs = ynew)
        }
        out$outsampresults = cbind(out$outsampresults, x2)
    }
    if (!is.null(ynew)) {
        out$outsampfitstats = c(R2 = getR2(ynew, ypred), rmse = sqrt(mean((ynew - 
            ypred)^2, na.rm = T)))
        if (length(unique(popnew)) > 1) {
            out$outsampfitstatspop = getR2pop(ynew, ypred, popnew)
            R2centered = getR2pop(ynew, ypred, popnew, type = "centered")
            R2scaled = getR2pop(ynew, ypred, popnew, type = "scaled")
            out$outsampfitstats = c(out$outsampfitstats, R2centered = R2centered, 
                R2scaled = R2scaled)
        }
    }
    if (returnGPgrad) {
        out$GPgrad = as.data.frame(gpgrad)
    }
    class(out) = "GPpred"
    return(out)
}
<bytecode: 0x31860b58>
<environment: namespace:GPEDM>


#### summary.GP ####
summary.GP <- function (object, ...) 
{
    d = ncol(object$inputs$X)
    cat("Number of predictors:", d, "\n")
    if (!is.null(object$b)) {
        if (length(object$b) == 1) {
            cat("Fisheries model b:", round(object$b, 5), "\n")
        }
        else {
            cat("Fisheries model b:\n")
            print.data.frame(data.frame(b = object$b), digits = 5)
        }
    }
    cat("Length scale parameters:\n")
    if (!is.null(object$inputs$x_names2)) {
        print.data.frame(data.frame(predictor = object$inputs$x_names2, 
            posteriormode = round(object$pars[1:d], 5)))
    }
    else if (!is.null(object$inputs$x_names)) {
        print.data.frame(data.frame(predictor = object$inputs$x_names, 
            posteriormode = round(object$pars[1:d], 5)))
    }
    else {
        print.data.frame(data.frame(posteriormode = round(object$pars[1:d], 
            5)))
    }
    cat("Process variance (ve):", object$pars["ve"])
    cat("\nPointwise prior variance (sigma2):", object$pars["sigma2"])
    np = length(unique(object$inputs$pop))
    cat("\nNumber of populations:", np)
    if (np > 1) {
        if (!is.null(object$inputs$rhomatrix)) {
            cat("\nDynamic correlation (rho):", "rhomatrix")
        }
        else {
            cat("\nDynamic correlation (rho):", object$pars["rho"])
        }
    }
    cat("\nIn-sample R-squared:", object$insampfitstats["R2"], 
        "\n")
    if (np > 1) {
        cat("In-sample R-squared by population:\n")
        print.data.frame(data.frame(R2 = object$insampfitstatspop$R2))
    }
    if (!is.null(object$outsampfitstats)) {
        cat("Out-of-sample R-squared:", object$outsampfitstats["R2"])
        if (np > 1) {
            cat("\nOut-of-sample R-squared by population:\n")
            print.data.frame(data.frame(R2 = object$outsampfitstatspop$R2))
        }
    }
}
<bytecode: 0x319263c8>
<environment: namespace:GPEDM>


#### ytransfun ####
ytransfun <- function (y, m1, e1, ytrans) 
{
    if (ytrans == "none") 
        return(y)
    if (ytrans == "log") 
        return(log(y))
    if (ytrans == "gr1") 
        return(log(y/m1))
    if (ytrans == "gr2") 
        return(log(y/e1))
}
<bytecode: 0x31939170>
<environment: namespace:GPEDM>


#### ytransfuninv ####
ytransfuninv <- function (y, m1, e1, ytrans) 
{
    if (ytrans == "none") 
        return(y)
    if (ytrans == "log") 
        return(exp(y))
    if (ytrans == "gr1") 
        return(exp(y) * m1)
    if (ytrans == "gr2") 
        return(exp(y) * e1)
}
<bytecode: 0x31945940>
<environment: namespace:GPEDM>
