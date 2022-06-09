# This script takes in a file with CosmicTrackDiag TTrees
# and calibrates the longitudinal reconstruction
# Including tdCentralRes, tdResSlope, eHalfPVScale, strawHalfPropVelocity
# and delta t time offsets in TrkPreampStraw

import ROOT
import numpy as np
import iminuit
# import matplotlib.pyplot as plt
import sys

commit_db = False
iov = ""
use_text_db = False
db_filename = ""
db_purpose = ""
db_version = ""
db_run = 0

input_filename = ""
output_dbfilename = ""
output_fclfilename = ""

if len(sys.argv) >= 6:
  input_filename = sys.argv[1]
  output_filename = sys.argv[2]
  output_fclfilename = sys.argv[3]
  commit_db = int(sys.argv[4])
  iov = sys.argv[5]
  if len(sys.argv) == 7:
    use_text_db = True
    db_filename = sys.argv[6]
  else:
    db_purpose = sys.argv[7]
    db_version = sys.argv[8]
    db_run = int(sys.argv[9])
    if len(sys.argv) > 9:
      db_subrun = int(sys.argv[9])
    else:
      db_subrun = 0
else:
  input_filename = input("ROOT file to process: ")
  output_dbfilename = input("Output database txt filename: ")
  output_fclfilename = input("Output fcl txt filename: ")
  commit_db = True if input("Commit results to database? y/[n] ") == "y" else False
  iov = input("iov for output database table: ")
  use_text_db = True if input("Use text file for existing table? y/[n] ") == "y" else False
  if use_text_db:
    db_filename = input("Filename for input table: ")
  else:
    db_purpose = input("Db purpose for input table: ")
    db_version = input("Db version for input table: ")
    db_run = int(input("Run number for input table: "))
    db_subrun = int(input("sub run number for input table: "))

import DbService
dbt = DbService.DbTool()
dbt.init()
if not use_text_db:
  cid = dbt.findCid(db_purpose,db_version,"TrkPreampStraw",db_run,db_subrun)
  temp = dbt.returnContent(cid).split("\n")
else:
  fdb = open(db_filename)
  line = fdb.readline()
  while line:
    if line.strip().startswith("TABLE") and "TrkPreampStraw" in line:
      break
    line = fdb.readline()
  temp = []
  line = fdb.readline()
  while line:
    if line.strip().startswith("#"):
      continue
    if line.strip().startswith("TABLE"):
      break
    temp.append(line)
    line = fdb.readline()
current_table = []
for i in range(len(temp)):
  data = temp[i].split(",")
  if len(data) == 6:
    current_table.append([int(data[0]),float(data[1]),float(data[2]),float(data[3]),float(data[4]),float(data[5])])


print("Processing root file:")
data = []

f = ROOT.TFile(input_filename)
t = f.Get("CosmicTrackDiag/hitT")

for i in range(t.GetEntries()):
  if i % 100000 == 0:
    print(i,t.GetEntries(),i/float(t.GetEntries())*100)
  t.GetEntry(i)
  if not (t.plane%4) in [0,3]:
    continue
  if not t.hitused or np.abs(t.ublong) > t.strawlen or t.tcnhits != t.ontrackhits:
    continue
  data.append([t.straw,t.panel,t.strawlen,t.ublong,t.deltat,t.edep,t.PanelsCrossedInEvent,t.tcnhits])
f.close()

print("Done processing. Starting energy fit:")


# GET ENERGY INFO
n_ebins = 24
w_ebin = 0.25
energies = [w_ebin*(i+0.5) for i in range(n_ebins)]
n_priorbins = 38
lmax = 380
n_lbins = 152
w_priorbins = lmax*2/n_priorbins
n_tbins = 40
tmax = 5


# get longitudinal position prior for each energy bin
scalings = []
hlpriors = [ROOT.TH1F("hu%d" % i,"hu%d" % i,n_priorbins,-1*lmax,lmax) for i in range(n_ebins)]
h2s = [ROOT.TH2F("h2%d" % i,"h2%d" % i,n_tbins,-1*tmax,tmax,152,-1*lmax,lmax) for i in range(n_ebins)]

for i in range(len(data)):
  # for energy fits we sum all straws < 60, and cut at position of 380 to make sure we aren't including effects from ends
  if data[i][0] < 60 and np.abs(data[i][3]) < lmax:
    ebin = min(n_ebins-1,int(data[i][5]/w_ebin))
    hlpriors[ebin].Fill(data[i][3])
    h2s[ebin].Fill(data[i][4],data[i][3])
for i in range(len(hlpriors)):
  hscale = ROOT.TH1F("hs%d" % i,"hs%d" % i,n_priorbins,-1*lmax,lmax)
  expected = hlpriors[i].Integral()/float(n_priorbins)
  for j in range(hlpriors[i].GetNbinsX()):
    hscale.SetBinContent(j+1,expected/max(hlpriors[i].GetBinContent(j+1),1))
  scalings.append([hscale.GetBinContent(j+1) for j in range(n_priorbins)])

ht = ROOT.TH1F("ht","ht",h2s[0].GetNbinsY(),h2s[0].GetYaxis().GetBinLowEdge(1),h2s[0].GetYaxis().GetBinLowEdge(h2s[0].GetNbinsY()+1))
f1 = ROOT.TF1("f1","gaus",-600,600)

totals = [[h2s[j].Integral(i+1,i+1,1,-1)+1e-15 for i in range(h2s[j].GetNbinsX())] for j in range(len(h2s))]
deltats = [h2s[0].GetXaxis().GetBinCenter(i+1) for i in range(h2s[0].GetNbinsX())]

current_ebin = 0

def fcn(speed,w0,w1):
  llike = 0

  for i in range(len(deltats)):
    if totals[current_ebin][i] == 0:
      continue
    f1.SetParameter(1,deltats[i]*speed)
    f1.SetParameter(2,(w0 + w1*(deltats[i]*speed)**2))
    f1.SetParameter(0,1)
    ttotal = 1e-10
    for j in range(ht.GetNbinsX()):
      bin_start = ht.GetXaxis().GetBinLowEdge(j+1)
      bin_end = ht.GetXaxis().GetBinLowEdge(j+2)
      iscale = int(max(min(n_priorbins-1,((bin_start+bin_end)/2./w_priorbins)+n_priorbins/2),0))
      scaling = scalings[current_ebin][iscale]
      val = (f1.Eval(bin_start)+f1.Eval(bin_end))/2./scaling
      ht.SetBinContent(j+1,val)
      ttotal += val

    for j in range(ht.GetNbinsX()):
      n_i = h2s[current_ebin].GetBinContent(i+1,j+1)
      u_i = max(ht.GetBinContent(j+1)/ttotal,1e-4)
      llike -= n_i*np.log(u_i)
  return llike


speed = 111.
w0 = 50
w1 = 0.0002
m = iminuit.Minuit(fcn, speed, w0, w1)
m.errordef = 0.5
speeds = []
speederrs = []
w0s = []
w0errs = []
w1s = []
w1errs = []
print("Starting energy scaling fits:")
for i in range(len(scalings)):
  current_ebin = i
  m.migrad()
  speeds.append(m.values['speed'])
  speederrs.append(m.errors['speed'])
  w0s.append(m.values['w0'])
  w0errs.append(m.errors['w0'])
  w1s.append(m.values['w1'])
  w1errs.append(m.errors['w1'])
  print(m.values,m.errors)

# plt.figure()
# plt.errorbar(x=energies,y=speeds,yerr=speederrs)
# plt.figure()
# plt.errorbar(x=energies,y=w0s,yerr=w0errs)
# plt.figure()
# plt.errorbar(x=energies,y=w1s,yerr=w1errs)
# plt.show()

base_speed = speeds[4]
base_w0 = w0s[4]
base_w1 = w1s[4]

speederrs = [speederrs[i]/speeds[4] for i in range(len(speeds))]
speeds = [speeds[i]/speeds[4] for i in range(len(speeds))]
w0rrs = [w0errs[i]/w0s[4] for i in range(len(w0s))]
w0s = [w0s[i]/w0s[4] for i in range(len(w0s))]
w1rrs = [w1errs[i]/w1s[4] for i in range(len(w1s))]
w1s = [w1s[i]/w1s[4] for i in range(len(w1s))]
#print(speeds)
#print(speederrs)
#print(w0s)
#print(w0errs)
#print(w1s)
#print(w1errs)


print("Starting per channel delay and propagation velocity fits:")

lmax = 0.9
n_priorbins = 36
tmax = 10
n_tbins = 160
nlbins = 180

scalings = []
escalings = []
strawlens = [0 for i in range(6*96)]
hlpriors = [ROOT.TH1F("hu%d_2" % i,"hu%d" % i,n_priorbins,-1*lmax,lmax) for i in range(6*96)]
h2s = [ROOT.TH2F("h2%d_2" % i,"h2%d" % i,n_tbins,-1*tmax,tmax,nlbins,-1*lmax,lmax) for i in range(6*96)]
hes = [ROOT.TH1F("hes%d_2" % i,"hes%d" % i,n_ebins,0,w_ebin*(n_ebins)) for i in range(6*96)]
hdt0s = [ROOT.TH1F("hdt0%d_2" % i,"hdt0%d" % i,100,-1*tmax,tmax) for i in range(6*96)]

#  fout.write("%d %d %f %f %f %f\n" % (t.straw,t.panel,t.strawlen,t.ublong,t.deltat,t.edep))
for i in range(len(data)):
  if np.abs(data[i][3])/data[i][2] < lmax and int(data[i][6]) == 1:
    straw = int(data[i][0]) + int(data[i][1])*96
    hlpriors[straw].Fill(data[i][3]/data[i][2])
    h2s[straw].Fill(data[i][4],data[i][3]/data[i][2])
    hes[straw].Fill(data[i][5])
    hdt0s[straw].Fill(data[i][4])
    strawlens[straw] = data[i][2]
for i in range(len(hlpriors)):
  hscale = ROOT.TH1F("hs%d" % i,"hs%d" % i,hlpriors[i].GetNbinsX(),hlpriors[i].GetBinLowEdge(1),hlpriors[i].GetBinLowEdge(hlpriors[i].GetNbinsX()+1))
  expected = hlpriors[i].Integral()/hlpriors[i].GetNbinsX()
  for j in range(hlpriors[i].GetNbinsX()):
    hscale.SetBinContent(j+1,expected/max(hlpriors[i].GetBinContent(j+1),1))
  scalings.append([hscale.GetBinContent(j+1) for j in range(hscale.GetNbinsX())])
  hes[i].Scale(1.0/max(1,hes[i].Integral()))
  escalings.append([hes[i].GetBinContent(j+1) for j in range(hes[i].GetNbinsX())])


ht = ROOT.TH1F("ht_2","ht",h2s[0].GetNbinsY(),h2s[0].GetYaxis().GetBinLowEdge(1),h2s[0].GetYaxis().GetBinLowEdge(h2s[0].GetNbinsY()+1))

totals = [[h2s[j].Integral(i+1,i+1,1,-1)+1e-15 for i in range(h2s[j].GetNbinsX())] for j in range(len(h2s))]
deltats = [h2s[0].GetXaxis().GetBinCenter(i+1) for i in range(h2s[0].GetNbinsX())]

current_straw = 0

def fcn(speed,offset,w0,w1):
  llike = 0

  for i in range(len(deltats)):
    if totals[current_straw][i] == 0:
      continue
    ht.Scale(0)
    ttotal = 1e-10
    tbins = []
    f1.SetParameter(1,(deltats[i]+offset)*speed/strawlens[current_straw])
    f1.SetParameter(2,(w0 + w1*((deltats[i]+offset)*speed)**2)/strawlens[current_straw])
    f1.SetParameter(0,1)
    for j in range(ht.GetNbinsX()):
      bin_start = ht.GetXaxis().GetBinLowEdge(j+1)
      bin_end = ht.GetXaxis().GetBinLowEdge(j+2)
      iscale = int(max(min(n_priorbins-1,((bin_start+bin_end)/2./w_priorbins)+n_priorbins/2),0))
      scaling = scalings[current_straw][iscale]
      val = (f1.Eval(bin_start)+f1.Eval(bin_end))/2./scaling
      tbins.append(val)
      ttotal += val
    for j in range(ht.GetNbinsX()):
      ht.SetBinContent(j+1,ht.GetBinContent(j+1) + tbins[j]/ttotal)

    for j in range(ht.GetNbinsX()):
      n_i = h2s[current_straw].GetBinContent(i+1,j+1)
      u_i = max(ht.GetBinContent(j+1),1e-4)
      llike -= n_i*np.log(u_i)
#  print(speed,offset,w0,w1,llike)
  return llike



speed = 111.
offset = 0
w0 = 50
w1 = 0.0002
m = iminuit.Minuit(fcn, speed, offset,w0, w1)
m.errordef = 0.5

straw_speeds = []
straw_speederrs = []
straw_w0s = []
straw_w0errs = []
straw_w1s = []
straw_w1errs = []
straw_offsets = []
straw_offseterrs = []
hdto = ROOT.TH1F("hdto","hdto",100,-10,10)


for i in range(len(scalings)):
  current_straw = i

  average_speed_scaling = 0
  average_w0_scaling = 0
  average_w1_scaling = 0
  for j in range(len(escalings[i])):
    average_speed_scaling += speeds[j]*escalings[i][j]
    average_w0_scaling += w0s[j]*escalings[i][j]
    average_w1_scaling += w1s[j]*escalings[i][j]
  speed = base_speed * average_speed_scaling
  w0 = base_w0 * average_w0_scaling
  w1 = base_w1 * average_w1_scaling

  m.values['speed'] = speed
  m.values['offset'] = -1*hdt0s[i].GetMean()
  m.values['w0'] = w0
  m.values['w1'] = w1
  m.fixed['w0'] = True
  m.fixed['w1'] = True

  if h2s[i].Integral() > 10:
    m.migrad()
    corrected_speed = m.values['speed'] / average_speed_scaling
  else:
    corrected_speed = 100.0
    average_speed_scaling = 1.0
  print(i,corrected_speed,m.errors['speed'],"  ",m.values['offset'],m.errors['offset'],"  ",m.values['w0'],m.errors['w0'],"  ",m.values['w1'],m.errors['w1'])


  straw_speeds.append(corrected_speed)
  straw_speederrs.append(m.errors['speed'] / average_speed_scaling)
  straw_offsets.append(m.values['offset'])
  straw_offseterrs.append(m.errors['offset'])
  straw_w0s.append(m.values['w0'])
  straw_w0errs.append(m.errors['w0'])
  straw_w1s.append(m.values['w1'])
  straw_w1errs.append(m.errors['w1'])

straws = [i for i in range(len(straw_speeds))]
# plt.figure()
# plt.errorbar(x=straws,y=straw_speeds,yerr=straw_speederrs)
# plt.figure()
# plt.errorbar(x=straws,y=straw_offsets,yerr=straw_offseterrs)
# plt.figure()
# plt.errorbar(x=straws,y=straw_w0s,yerr=straw_w0errs)
# plt.figure()
# plt.errorbar(x=straws,y=straw_w1s,yerr=straw_w1errs)
# plt.show()

print(straw_speeds)
print(straw_speederrs)
print(straw_offsets)
print(straw_offseterrs)
print(straw_w0s)
print(straw_w0errs)
print(straw_w1s)
print(straw_w1errs)

fout = open(output_fclfilename,"w")
fout.write("services.ProditionsService.strawResponse.eBins : %d\n" % n_ebins)
fout.write("services.ProditionsService.strawResponse.eBinWidth : %.2f\n" % w_ebin)
fout.write("services.ProditionsService.strawResponse.eHalfPVScale : [")
for i in range(n_ebins):
  fout.write(" %7.4f" % speeds[i])
  if i != n_ebins-1:
    fout.write(",")
fout.write("]\n")
fout.write("services.ProditionsService.strawResponse.tdCentralRes : [")
for i in range(n_ebins):
  fout.write(" %.3f" % (base_w0*w0s[i]))
  if i != n_ebins-1:
    fout.write(",")
fout.write("]\n")
fout.write("services.ProditionsService.strawResponse.tdResSlope : [")
for i in range(n_ebins):
  fout.write(" %.6f" % (base_w1*w1s[i]))
  if i != n_ebins-1:
    fout.write(",")
fout.write("]\n")
fout.write("services.ProditionsService.strawResponse.strawHalfPropVelocity : [")
for i in range(len(straw_speeds)):
  fout.write(" %.3f" % straw_speeds[i])
  if i != len(straw_speeds)-1:
    fout.write(",")
fout.write("]\n")
fout.close()

for i in range(len(straw_offsets)):
  current_table[i][1] += straw_offsets[i]/2.
  current_table[i][2] -= straw_offsets[i]/2.

fout = open(output_dbfilename,"w")
fout.write("TABLE TrkPreampStraw")
if iov:
  fout.write(" %s\n" % iov)
else:
  fout.write("\n")
for i in range(len(current_table)):
  fout.write("%d, %f, %f, %f, %f, %f\n" % (current_table[i][0],current_table[i][1],current_table[i][2],current_table[i][3],current_table[i][4], current_table[i][5])) 
fout.close()

if commit_db:
  if iov:
    dbt.setArgs(["commit-calibration","--file",output_dbfilename,"-addIOV"])
  else:
    dbt.setArgs(["commit-calibration","--file",output_dbfilename])
  dbt.run()

print("Finished")
