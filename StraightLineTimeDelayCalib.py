# This script takes in a file with CosmicTrackDiag TTrees
# and calibrates the drift reconstruction
# currently only adjusts
# absolute time offsets in TrkPreampStraw

import ROOT
import numpy as np
import sys
import tempfile

import DbTables
import DbService

commit_db = False
iov = ""
use_text_db = False
db_filename = ""
db_purpose = ""
db_version = ""
db_run = 0

input_filename = ""
output_filename = ""


if len(sys.argv) >= 6:
  input_filename = sys.argv[1]
  output_filename = sys.argv[2]
  commit_db = int(sys.argv[3])
  iov = sys.argv[4]
  if len(sys.argv) == 6:
    use_text_db = True
    db_filename = sys.argv[5]
  else:
    db_purpose = sys.argv[5]
    db_version = sys.argv[6]
    db_run = int(sys.argv[7])
    if len(sys.argv) > 8:
      db_subrun = int(sys.argv[8])
    else:
      db_subrun = 0
else:
  input_filename = input("ROOT file to process: ")
  output_filename = input("Output database txt filename: ")
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

if not use_text_db:
  dbt = DbService.DbTool()
  dbt.init()
  cid = dbt.findCid(db_purpose,db_version,"TrkPreampStraw",db_run,db_subrun)
  temp = dbt.returnContent(cid).split("\n")
else:
  dbu = DbTables.DbUtil()
  dtc = dbu.readFile(db_filename)
  for dt in dtc:
    if dt.table().name() == "TrkPreampStraw":
      t = dt.table()
      t.toCsv()
      temp = t.csv().split("\n")
      break

current_table = []
for i in range(len(temp)):
  data = temp[i].split(",")
  if len(data) == 6:
    current_table.append([int(data[0]),float(data[1]),float(data[2]),float(data[3]),float(data[4]),float(data[5])])


f = ROOT.TFile(input_filename)
t = f.Get("CosmicTrackDiag/hitT")

h2 = ROOT.TH2F("h2","h2",96*6,0,96*6,240,-60,60)

t.Project("h2","ubtresid:straw+panel*96","hitused && abs(ubtresid) < 60 && ontrackhits==tcnhits && abs(ubdoca) < 2.5")

hp = h2.ProfileX("hp")

hp.Draw()
input()

for i in range(96*6):
  current_table[i][1] -= hp.GetBinContent(i+1)
  current_table[i][2] -= hp.GetBinContent(i+1)

if output_filename:
  fout = open(output_filename,"w")
else:
  handle, output_filename = tempfile.mkstemp()
  fout = os.fdopen(handle)
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
    dbt.setArgs(["commit-calibration","--file",output_filename,"-addIOV"])
  else:
    dbt.setArgs(["commit-calibration","--file",output_filename])
  dbt.run()

print("Finished")
