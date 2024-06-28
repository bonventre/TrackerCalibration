import uproot
import awkward as ak
import numpy as np
import ROOT
import matplotlib.pyplot as plt
import sys

inputfiles = sys.argv[1]
dbfilein = sys.argv[2]
dbfileout = sys.argv[3]
fclfileout = sys.argv[4]
iteration = int(sys.argv[5])

halflengths = np.array([587.878, 585.846, 583.79, 581.711, 579.607, 577.478, 575.325, 573.146, 570.942, 568.713, 566.457, 564.175, 561.866, 559.531, 557.168, 554.777, 552.359, 549.912, 547.436, 544.931, 542.396, 539.832, 537.237, 534.611, 531.954, 529.265, 526.544, 523.79, 521.003, 518.182, 515.326, 512.436, 509.51, 506.547, 503.548, 500.512, 497.437, 494.324, 491.17, 487.977, 484.742, 481.465, 478.146, 474.782, 471.374, 467.921, 464.42, 460.872, 457.275, 453.627, 449.929, 446.178, 442.373, 438.513, 434.596, 430.62, 426.585, 422.488, 418.328, 414.102, 409.809, 405.447, 401.012, 396.504, 391.918, 387.253, 382.506, 377.673, 372.752, 367.737, 362.627, 357.416, 352.101, 346.676, 341.135, 335.475, 329.687, 323.766, 317.704, 311.493, 305.123, 298.585, 291.866, 284.955, 277.838, 270.496, 262.913, 255.067, 246.931, 238.477, 229.67, 220.467, 210.817, 200.654, 189.897, 178.438])


abst = np.zeros(36*6*96)
deltat = np.zeros(36*6*96)
propv = np.zeros(36*6*96)
totdt = np.ones(16*5)*20.0
totdterr = np.ones(16*5)*11.0
ecor = np.zeros(59)
ecordt = np.zeros(59)


fields = ["udt","utresid","plane","panel","straw","uupos","wdist","udoca","etime","state","edep","tot"]
a = {field: [] for field in fields}
for batch in uproot.iterate(inputfiles,filter_name=["kl.status","kltsh"]):
    print(batch)
    cut = ak.sum(batch["kl.status"],axis=1) == 1
    for field in fields:
        a[field].append(ak.flatten(ak.flatten(batch["kltsh"][field][cut])).to_numpy())
for field in fields:
    a[field] = np.concatenate(a[field])

f1 = ROOT.TF1("pol3","pol3")
for iplane in range(36):
    cut0 = (a["plane"] == iplane)
    etime0 = a["etime"][cut0]
    uupos0 = a["uupos"][cut0]
    udt0   = a["udt"][cut0]
    utresid0 = a["utresid"][cut0]
    panel0 = a["panel"][cut0]
    straw0 = a["straw"][cut0]
    state0 = a["state"][cut0]
    for ipanel in range(6):
        cut1 = (panel0 == ipanel)
        etime1 = etime0[cut1]
        uupos1 = uupos0[cut1]
        udt1   = udt0[cut1]
        utresid1 = utresid0[cut1]
        straw1 = straw0[cut1]
        state1 = state0[cut1]
        print("Plane/panel",iplane,ipanel)
        for istraw in range(96):
            cut2 = (straw1 == istraw)
            etime2 = etime1[cut2]
            uupos2 = uupos1[cut2]
            udt2 = udt1[cut2]
            utresid2 = utresid1[cut2]
            state2 = state1[cut2]
            if iteration == 0:
                cut3 = (udt2 < 100) & (udt2 > -30) & (np.abs(state2) == 1) & (np.abs(uupos2) < 600)
                etime3 = etime2[cut3]
                uupos3 = uupos2[cut3]
                udt3 = udt2[cut3]
                abst[iplane*6*96+ipanel*96+istraw] = np.mean(udt3) - 20.0
            else:
                cut3 = np.abs(utresid2) < 30
                etime3 = etime2[cut3]
                uupos3 = uupos2[cut3]
                utresid3 = utresid2[cut3]
                abst[iplane*6*96+ipanel*96+istraw] = np.mean(utresid3)

            if np.isnan(abst[iplane*6*96+ipanel*96+istraw]):
                abst[iplane*6*96+ipanel*96+istraw] = 0

            x = etime3[:,1] - etime3[:,0]
            y = uupos3
            if len(x) < 100:
                print("NOT ENOUGH:",iplane,ipanel,istraw,":",len(x))
                continue
            h2 = ROOT.TH2F("","h2",200,-20,20,600,-600,600)
            mv = np.mean(x)
            h2.FillN(len(x),x.astype(np.double),y.astype(np.double),np.ones(x.shape,dtype=np.double))
            fr = h2.Fit(f1,"0QSR","",mv-5,mv+5)
            for k in range(20):
                t = mv-1.0+0.1*k
                p1 = f1.Eval(t)
                if p1 > 0 or k == 19:
                    p0 = f1.Eval(t-0.1)
                    x0 = t-0.1
                    x1 = t
                    slope = (p1-p0)/(x1-x0)
                    intercept = p0 - slope * x0
                    break
            mv = -1*intercept/slope
            fr = h2.Fit(f1,"0QSR","",mv-5,mv+5)
            for k in range(20):
                t = mv-1.0+0.1*k
                p1 = f1.Eval(t)
                if p1 > 0 or k == 19:
                    p0 = f1.Eval(t-0.1)
                    x0 = t-0.1
                    x1 = t
                    slope = (p1-p0)/(x1-x0)
                    intercept = p0 - slope * x0
                    break
            deltat[iplane*6*96+ipanel*96+istraw] = intercept/slope
            propv[iplane*6*96+ipanel*96+istraw] = slope

sid = a["plane"]*6*96 + a["panel"]*96 + a["straw"]
udt = a["udt"] - abst[sid]
tot = (a["tot"][:,1]+a["tot"][:,0])/2.
for j in range(16):
    print("tot bin",j)
    for i in range(5):
        lbound = 0.0005*i
        ubound = 0.0005*(i+1)
        if i == 4:
            ubound = 999.
        cut = (a["state"] > -2) & (np.abs(a["uupos"]) < halflengths[a["straw"]]*1.0) & (a["edep"] > lbound) & (a["edep"] < ubound) & (udt < 40) & (udt > -40) & (tot == j*5)
        dtimes = udt[cut]
        if len(dtimes) < 100:
            continue
        totdt[j*5+i] = np.mean(dtimes)
        totdterr[j*5+i] = np.std(dtimes)



for i in range(59):
  print("deltat bin",i)
  cut = (a["state"] > -2) & (np.abs(a["uupos"]) < halflengths[a["straw"]]*1.0) & ( a["edep"] > 0.0001*i) & (a["edep"] < 0.0001*(i+1))
  x = a["etime"][cut][:,1]-a["etime"][cut][:,0]
  y = a["uupos"][cut]
  sid = a["plane"][cut]*6*96 + a["panel"][cut]*96 + a["straw"][cut]
  yc = (x+deltat[sid])*(propv[sid]+1e-4)

  cut2 = np.abs(yc) < halflengths[a["straw"][cut]]*0.5
  y = y[cut2]
  yc = yc[cut2]

  if len(x) < 100:
    ecor[i] = 1.0
    ecordt[i] = 0.0
    continue
  
  hp = ROOT.TProfile("","hp",100,-600,600)
  hp.FillN(len(y),yc.astype(np.double),y.astype(np.double),np.ones(y.shape,dtype=np.double))
  fr = hp.Fit("pol1","0QS")
  ecor[i] = fr.GetParams()[1]
  ecordt[i] = fr.GetParams()[0]

#import pdb;pdb.set_trace()
fin = open(dbfilein)
lines = fin.readlines()
i = 0
while True:
    if i >= len(lines):
        sys.exit(1)
    if "TABLE TrkPreampStraw" in lines[i]:
        break
    i += 1

fout = open(dbfileout,"w")
fout.write(lines[i])
i += 1
for j in range(20736):
    ij = i+j
    parts = lines[ij].split(",")
    delayhv = float(parts[1])
    delaycal = float(parts[2])
    threshhv = float(parts[3])
    threshcal = float(parts[4])
    gain = float(parts[5])

    delayhv -= abst[j]
    delaycal -= abst[j]
    delayhv += deltat[j]/2.
    delaycal -= deltat[j]/2.
    fout.write("%6s,%7.3f,%7.3f,%7.3f,%7.3f,%10.1f\n" % (parts[0],delayhv,delaycal,threshhv,threshcal,gain))
fout.close()

fout = open(fclfileout,"w")
ebins = 59
ebinwidth = 0.1
lres = 80.0
totebins = 5
totebinwidth = 0.0005
fout.write("services.ProditionsService.strawResponse.eBins : %d\n" % ebins)
fout.write("services.ProditionsService.strawResponse.eBinWidth : %f\n" % ebinwidth)
fout.write("services.ProditionsService.strawResponse.eHalfPVScale : [ ")
for j in range(ebins):
    fout.write("%5.3f" % ecor[j])
    if j != ebins-1:
        fout.write(", ")
fout.write(" ]\n")
fout.write("services.ProditionsService.strawResponse.tdCentralRes : [ ")
for j in range(ebins):
    fout.write("%5.3f" % lres)
    if j != ebins-1:
        fout.write(", ")
fout.write(" ]\n")
fout.write("services.ProditionsService.strawResponse.tdResSlope : [ ")
for j in range(ebins):
    fout.write("%5.3f" % 0.0)
    if j != ebins-1:
        fout.write(", ")
fout.write(" ]\n")
fout.write("services.ProditionsService.strawResponse.totTBins : 16\n")
fout.write("services.ProditionsService.strawResponse.totEBins : %d\n" % totebins)
fout.write("services.ProditionsService.strawResponse.totEBinWidth : %f\n" % totebinwidth)
fout.write("services.ProditionsService.strawResponse.totDriftTime : [\n")
for j in range(16):
    for k in range(totebins):
        fout.write("%5.2f" % totdt[j*totebins+k])
        if j*totebins+k != 16*totebins-1:
            fout.write(", ")
    fout.write("\n")
fout.write(" ]\n")
fout.write("services.ProditionsService.strawResponse.totDriftError : [\n")
for j in range(16):
    for k in range(totebins):
        fout.write("%5.2f" % totdterr[j*totebins+k])
        if j*totebins+k != 16*totebins-1:
            fout.write(", ")
    fout.write("\n")
fout.write(" ]\n")
fout.write("services.ProditionsService.strawResponse.strawHalfPropVelocity : [ ")
for j in range(20736):
    if propv[j] < 10.0:
        propv[j] = 100.0
    fout.write("%5.1f" % propv[j])
    if j != 20735:
        fout.write(", ")
fout.write(" ]\n")
fout.close()

