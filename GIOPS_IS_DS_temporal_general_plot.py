# Import needed modules
import numpy as np
import datetime 
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter,NullLocator
import matplotlib.dates as mdates
import sys, os, errno
import time
from scipy import stats
from matplotlib.patches import Rectangle

initial_time = time.time()

# Define functions
def read_file(path, suite, date, obs):
        '''
	function to read sst statistic data from SAM/DIA file
  	Input: analysis date
         suite name  
         suite path
	Output: Numpy array containing the statistics 
        '''
	file = path + suite + '/' + date + '/DIA/' + date + '00_SAM.dia_' + obs
	return pd.read_csv(file, skiprows=2, delim_whitespace=True, na_values=['-0.17977+309'], usecols=(3, 5, 7))
	

def make_fig():
	'''
	setup figure 
	'''
	fig = plt.figure(figsize=(5, 5))
	plt.tick_params(axis='x', which='major', labelsize=10)
	gs  = gridspec.GridSpec(3, 1, hspace = 0.5, wspace = 0.1, top = 0.9, left = 0.08, right = 0.95, bottom = 0.05)
	ax1 = fig.add_subplot(gs[0, 0])
	ax2 = fig.add_subplot(gs[1, 0])
	ax3 = fig.add_subplot(gs[2, 0])
	return ax1, ax2, ax3
		

def filter_zeros_avg(df):
	'''
	Filter dates with zeros data number and compute columns means
	input: pandas dataframe
	'''
	cols = df.columns.get_values().tolist()
	df_m = df
	df_m = df.replace({0:np.nan})
  	return df_m[cols].mean()


def plot_mean_stat(data, data_ref, obs, cycle, cycle_ref, p):
	'''
	Function to plot statistics
	Input:
	data: data to plot (dataframe of each obs) 
	data_ref: reference data 
	'''
	if obs in ['ALTIKA', 'CRYOSAT2', 'JASON2', 'JASON3', 'JASON2N', 'HY2A', 'SENTINEL3A']:
		data[47:] = data[47:] * 100                       # convert to cm
		data_ref[47:] = data_ref[47:] * 100               # convert to cm

	y1 = np.arange(0, len(regions[1:23])*2, 2)
	y2 = np.arange(0, len(regions[23:])*2, 2)
	width1 = [0.5] * len(y1) 
	width2 = [0.5] * len(y2) 
	width3 = [1.0] * len(y1)
	width4 = [1.0] * len(y2) 
  	# Num data
  	fig1 = plt.figure(figsize=(8, 10)); ax = fig1.gca()
	ax.barh(y1, data[1:23], width1,  align='center', color='red')
	ax.barh(y1+width1, data_ref[1:23], width1,  align='center', color='blue')
	ax.set_yticks(y1+0.25)
	ax.set_ylim([-1, y1[-1]+1])
        ax.set_yticklabels(regions[1:23], fontsize=12)
	ax.set_title('Average NUM(DATA)', fontsize=16)
	ax.text(0, y1[-1] + 1.25, obs.replace("_", " "), size=14, ha="left")
	ax.legend((cycle, cycle_ref))
	if not np.isnan(data[0]):
		ax.set_xlabel('Global = %8d            Global(ref) = %8d' %(data[0], data_ref[0]))
	else:
		ax.set_xlabel('Global = Nan            Global(ref) = Nan')
	fig1.subplots_adjust(left=0.2)
	plt.tight_layout()
   	plt.savefig(output_path + '/' + suite + '/' + obs +'/' + obs + '_' + start_date + '_' + final_date + '_NUM_DATA_Mean_N.png', dpi=80)
	plt.close()

  	fig2 = plt.figure(figsize=(8, 10)); ax = fig2.gca()
	ax.barh(y2, data[23:47], width2,  align='center', color='red')
	ax.barh(y2+width2, data_ref[23:47], width2,  align='center', color='blue')
	ax.set_yticks(y2+0.25)
	ax.set_ylim([-1, y2[-1]+1])
        ax.set_yticklabels(regions[23:47], fontsize=12)
	ax.set_title('Average NUM(DATA)', fontsize=16)
	ax.text(0, y2[-1] + 1.25, obs.replace("_", " "), size=14, ha="left")
	ax.legend((cycle, cycle_ref))
	if not np.isnan(data[0]):
		ax.set_xlabel('Global = %8d            Global(ref) = %8d' %(data[0], data_ref[0]))
	else:
		ax.set_xlabel('Global = NaN            Global(ref) = NaN')
		
	fig2.subplots_adjust(left=0.2)
	plt.tight_layout()
  	plt.savefig(output_path + suite + '/' + obs +'/' + obs + '_' + start_date + '_' + final_date + '_NUM_DATA_Mean_S.png', dpi=80)
  	plt.close()
        
	# OMP
  	fig1 = plt.figure(figsize=(8, 10)); ax = fig1.gca() 
	ax.barh(y1, data[48:70], width1,  align='center', color='red')
	ax.barh(y1+width1, data_ref[48:70], width1,  align='center', color='blue')
	ax.set_yticks(y1+0.25)
  	ax.set_yticklabels(regions[1:23], fontsize=12)
	ax.set_title('Average AVR(MISFIT)', fontsize=16)
	if obs in ['GEN_SST', 'GEN_SST_night']:
		ax.set_xlim([-1, 1])
 		ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
  	ax.xaxis.set_minor_locator(MultipleLocator(0.1))
	else:
		ax.set_xlim([-6, 6])
		ax.xaxis.set_major_formatter(FormatStrFormatter('%1d'))
		ax.xaxis.set_minor_locator(MultipleLocator(0.5))

	ax.set_ylim([-1, y1[-1]+1])
	ax.text(ax.get_xlim()[0], y1[-1] + 1.25, obs.replace("_", " "), size=14, ha="left")
	ax.legend((cycle, cycle_ref))
	ax.axvline(x=0, linewidth=0.2, color='k')
	if not np.isnan(data[0]):
		ax.set_xlabel('Global = %.5f            Global(ref) = %.5f' %(data[47], data_ref[47]))
	else:
		ax.set_xlabel('Global = NaN            Global(ref) = NaN')
	fig1.subplots_adjust(left=0.2)
	plt.tight_layout()
  	plt.savefig(output_path + suite + '/' + obs +'/' + obs + '_' + start_date + '_' + final_date + '_AVR_MISFIT_Mean_N.png', dpi=80)
	plt.close()

  	fig2 = plt.figure(figsize=(8, 10)); ax = fig2.gca()
	ax.barh(y2, data[70:94], width2,  align='center', color='red')
	ax.barh(y2+width2, data_ref[70:94], width2,  align='center', color='blue')
	ax.set_yticks(y2+0.25)
	ax.set_ylim([-1, y2[-1]+1])
  	ax.set_yticklabels(regions[23:], fontsize=12)
	ax.set_title('Average AVR(MISFIT)', fontsize=16)
	if obs in ['GEN_SST', 'GEN_SST_night']:
		ax.set_xlim([-1, 1])
 		ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
  	ax.xaxis.set_minor_locator(MultipleLocator(0.1))
	else:
		ax.set_xlim([-6, 6])
		ax.xaxis.set_major_formatter(FormatStrFormatter('%1d'))
		ax.xaxis.set_minor_locator(MultipleLocator(0.5))
	ax.text(ax.get_xlim()[0], y2[-1] + 1.25, obs.replace("_", " "), size=14, ha="left")
	ax.legend((cycle, cycle_ref))
	ax.axvline(x=0, linewidth=0.2, color='k')
	if not np.isnan(data[0]):
		ax.set_xlabel('Global = %.5f            Global(ref) = %.5f' %(data[47], data_ref[47]))
	else:
		ax.set_xlabel('Global = NaN            Global(ref) = NaN')
	fig2.subplots_adjust(left=0.2)
	plt.tight_layout()
   	plt.savefig(output_path + suite + '/' + obs +'/' + obs + '_' + start_date + '_' + final_date + '_AVR_MISFIT_Mean_S.png', dpi=80)
	plt.close()

	#RMS
  	fig1 = plt.figure(figsize=(8, 10))
	ax1 = fig1.gca()
	ax2 = ax1.twiny()
	ax1.barh(y1, data[95:117], width1,  align='center', color='red')
	ax1.barh(y1+width1, data_ref[95:117], width1,  align='center', color='blue')
	ax1.set_yticks(y1+0.25)
  	ax1.set_yticklabels(regions[1:23], fontsize=12)
	ax1.set_title('Average RMS(MISFIT)', fontsize=16)
	if obs in ['GEN_SST', 'GEN_SST_night']:
		ax1.set_xlim([0, 2])
 		ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
  	ax1.xaxis.set_minor_locator(MultipleLocator(0.1))
		bb = np.array([-0.02] * len(data[95:117]))
		ax2.set_xlim([-1.5, 0])
	else:
		ax1.set_xlim([0, 30])
		ax1.xaxis.set_major_formatter(FormatStrFormatter('%1d'))
		ax1.xaxis.set_minor_locator(MultipleLocator(1.0))
		bb = np.array([-0.2]*len(data[95:117]))
		ax2.set_xlim([-15., 0])
		
	ax1.set_ylim([-1, y1[-1]+1])
	ax1.text(ax1.get_xlim()[0], y1[-1] + 1.25, obs.replace("_", " "), size=14, ha="left")
	ax1.legend((cycle, cycle_ref), loc='upper center')
	if not np.isnan(data[0]):
		ax1.set_xlabel('Global = %.5f            Global(ref) = %.5f' %(data[94], data_ref[94]))
	else:
		ax1.set_xlabel('Global = NaN            Global(ref) = NaN')
  	recs = ax2.barh(y1+0.25, bb, width3, align='center', edgecolor='none')
	ax2.xaxis.set_major_locator(NullLocator())
	for i in range(len(recs)):
		if p[i+1] <= 0.05 and (data[i+95] > data_ref[i+95]):
			recs[i].set_facecolor("#95d0fc")
		if p[i+1] <= 0.05 and (data[i+95] < data_ref[i+95]):
			recs[i].set_facecolor("#cf6275")
		if p[i+1] > 0.05 :
			recs[i].set_facecolor("#d8dcd6")
	
	if obs in ['GEN_SST', 'GEN_SST_night']:
		if p[0] <= 0.05 and (data[94] > data_ref[94]):
			ax1.add_patch(Rectangle((1.9, -3.2), 0.05, 1.0, facecolor='#95d0fc', edgecolor='none', zorder=6, clip_on=False))
		if p[0] <= 0.05 and (data[94] < data_ref[94]):
			ax1.add_patch(Rectangle((1.9, -3.2), 0.05, 1.0, facecolor='#cf6275', edgecolor='none', zorder=6, clip_on=False))
		if p[0] > 0.05 :
			ax1.add_patch(Rectangle((1.9, -3.2), 0.05, 1.0, facecolor="#d8dcd6", edgecolor='none', zorder=6, clip_on=False))
	
	if obs not in ['GEN_SST', 'GEN_SST_night']:
		if p[0] <= 0.05 and (data[94] > data_ref[94]):
			ax1.add_patch(Rectangle((29.0, -3.2), 1.0, 1.0, facecolor='#95d0fc', edgecolor='none', zorder=6, clip_on=False))
		if p[0] <= 0.05 and (data[94] < data_ref[94]):
			ax1.add_patch(Rectangle((29.0, -3.2), 1.0, 1.0, facecolor='#cf6275', edgecolor='none', zorder=6, clip_on=False))
		if p[0] > 0.05 :
			ax1.add_patch(Rectangle((29.0, -3.2), 1.0, 1.0, facecolor="#d8dcd6", edgecolor='none', zorder=6, clip_on=False))
		
	fig1.subplots_adjust(left=0.2)
	plt.tight_layout()
  	plt.savefig(output_path + suite + '/' + obs +'/' + obs + '_' + start_date + '_' + final_date + '_RMS_MISFIT_Mean_N.png', dpi=80)
	plt.close()

  	fig2 = plt.figure(figsize=(8, 10))
	ax1 = fig2.gca()
	ax2 = ax1.twiny()
	ax1.barh(y2, data[117:], width2,  align='center', color='red')
	ax1.barh(y2+width2, data_ref[117:], width2,  align='center', color='blue')
	ax1.set_yticks(y2+0.25)
  	ax1.set_yticklabels(regions[23:], fontsize=12)
	ax1.set_title('Average RMS(MISFIT)', fontsize=16)
	if obs in ['GEN_SST', 'GEN_SST_night']:
		ax1.set_xlim([0, 2])
 		ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
  	ax1.xaxis.set_minor_locator(MultipleLocator(0.1))
		bb = np.array([-0.02] * len(data[117:]))
		ax2.set_xlim([-1.5, 0])
	else:
		ax1.set_xlim([0, 30])
		ax1.xaxis.set_major_formatter(FormatStrFormatter('%1d'))
		ax1.xaxis.set_minor_locator(MultipleLocator(1.0))
		bb = np.array([-0.2]*len(data[117:]))
		ax2.set_xlim([-15., 0])
	ax1.set_ylim([-1, y2[-1]+1])
	ax1.text(ax1.get_xlim()[0], y2[-1] + 1.25, obs.replace("_", " "), size=14, ha="left")
	ax1.legend((cycle, cycle_ref),loc='upper center')
	if not np.isnan(data[0]):
		ax1.set_xlabel('Global = %.5f            Global(ref) = %.5f' %(data[94], data_ref[94]))
	else:
		ax1.set_xlabel('Global = NaN            Global(ref) = NaN')
  	recs = ax2.barh(y2+0.25, bb, width4, align='center', edgecolor='none')
	ax2.xaxis.set_major_locator(NullLocator())
	for i in range(len(recs)):
		if p[i+23] <= 0.05 and (data[i+117] > data_ref[i+117]):
			recs[i].set_facecolor("#95d0fc")
		if p[i+23] <= 0.05 and (data[i+117] < data_ref[i+117]):
			recs[i].set_facecolor("#cf6275")
		if p[i+23] > 0.05 :
			recs[i].set_facecolor("#d8dcd6")
	if obs in ['GEN_SST', 'GEN_SST_night']:
		if p[0] <= 0.05 and (data[94] > data_ref[94]):
			ax1.add_patch(Rectangle((1.9, -3.4), 0.05, 1.0, facecolor='#95d0fc', edgecolor='none', zorder=6, clip_on=False))
		if p[0] <= 0.05 and (data[94] < data_ref[94]):
			ax1.add_patch(Rectangle((1.9, -3.4), 0.05, 1.0, facecolor='#cf6275', edgecolor='none', zorder=6, clip_on=False))
		if p[0] > 0.05 :
			ax1.add_patch(Rectangle((1.9, -3.4), 0.05, 1.0, facecolor="#d8dcd6", edgecolor='none', zorder=6, clip_on=False))
	
	if obs not in ['GEN_SST', 'GEN_SST_night']:
		if p[0] <= 0.05 and (data[94] > data_ref[94]):
			ax1.add_patch(Rectangle((29.0, -3.5), 1.0, 1.0, facecolor='#95d0fc', edgecolor='none', zorder=6, clip_on=False))
		if p[0] <= 0.05 and (data[94] < data_ref[94]):
			ax1.add_patch(Rectangle((29.0, -3.5), 1.0, 1.0, facecolor='#cf6275', edgecolor='none', zorder=6, clip_on=False))
		if p[0] > 0.05 :
			ax1.add_patch(Rectangle((29.0, -3.5), 1.0, 1.0, facecolor="#d8dcd6", edgecolor='none', zorder=6, clip_on=False))
	fig2.subplots_adjust(left=0.2)
	plt.tight_layout()
  	plt.savefig(output_path + suite + '/' + obs +'/' + obs + '_' + start_date + '_' + final_date + '_RMS_MISFIT_Mean_S.png', dpi=80)
	plt.close()
	return


# Declarations ( informations on the suite)
#suite_ref = 'giops_222_bugfix/gx_errx3'                                                        # Reference experiment (suite) name (Dorina)
suite_ref = 'gx_004'                                                                            # Reference experiment (suite) name
suite='gx_005'                                                                                  # Experiment (suite) name
path='/space/hall1/sitestore/eccc/cmd/e/kch001/maestro_archives/GIOPS/SAM2/giops_dev/'          # experiment path
path_ref='/space/hall1/sitestore/eccc/cmd/e/kch001/maestro_archives/GIOPS/SAM2/giops_dev/'      # Reference experiment path 
output_path ='/space/hall1/sitestore/eccc/cmd/e/kch001/SAM2_diags/giops_dev/'                   # Output path

start_date='20170621'                                                                           # Initial date of the cycle
final_date='20170830'                                                                           # Initial date of the cycle



sdate = datetime.datetime.strptime(start_date, "%Y%m%d")                                                   
fdate = datetime.datetime.strptime(final_date, "%Y%m%d")                                                   
num_dates = (fdate - sdate).days                                                               # Number of dates during the cycle
dates = [sdate + datetime.timedelta(days = x) for x in range(0, num_dates + 7, 7)]             # List of dates of the cycle
region_num = [str(item).zfill(2) for item in range(0,47)]                                      # The 47 regions
list_obs = 'GEN_SST GEN_SST_night ALTIKA CRYOSAT2 JASON2 JASON3 JASON2N HY2A SENTINEL3A'.split(' ')       # Instruments (obs) list
regions = "Global,\
           Irminger Sea,\
           Iceland Basin,\
           Newfoundland-Iceland,\
           Yoyo Pomme,\
           Gulf Stream2,\
           Gulf Stream1 XBT,\
           North Medeira XBT,\
           Charleston tide,\
           Bermuda tide,\
           Gulf of Mexico,\
           Florida Straits XBT,\
           Puerto Rico XBT,\
           Dakar,\
           Cape Verde XBT,\
           Rio-La_Coruna Woce,\
           Belem XBT,\
           Cayenne tide,\
           Sao Tome tide,\
           XBT-Central SEC,\
           Pirata,\
           Rio-La Coruna,\
           Ascension tide,\
           Antarctic,\
           South Atlantic,\
           Falkland current,\
           South_Atl. gyre,\
           Angola,\
           Benguela current,\
           Aghulas Region,\
           Pacific Region,\
           North Pacific gyre,\
           California current,\
           North Tropical Pacific,\
           Nino1+2,\
           Nino3,\
           Nino4,\
           Nino6,\
           Nino5,\
           South Tropical Pacific,\
           South Pacific Gyre,\
           Peru coast,\
           Chile coast,\
           Eastern Australia,\
           Indian Ocean,\
           Tropical Indian Ocean,\
           South Indian Ocean"	
regions = " ".join(regions.split()).split(",")    # transform to list

# Create output path for every instrument
for i in range(len(list_obs)):
	try:
		os.makedirs(output_path + suite + '/' + list_obs[i])
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise
# Arrays initialization
count = np.zeros(shape=(len(list_obs), len(dates), len(regions)), dtype=float)
omp   = np.zeros(shape=(len(list_obs), len(dates), len(regions)), dtype=float)
ms    = np.zeros(shape=(len(list_obs), len(dates), len(regions)), dtype=float)

count_ref = np.zeros(shape=(len(list_obs), len(dates), len(regions)), dtype=int)
omp_ref   = np.zeros(shape=(len(list_obs), len(dates), len(regions)), dtype=float)
ms_ref    = np.zeros(shape=(len(list_obs), len(dates), len(regions)), dtype=float)

# Read data and store in numpy arrays
for num_dates in range(len(dates)):
        dateS = datetime.datetime.strftime(dates[num_dates], "%Y%m%d")
        for obs in list_obs:
		      print dateS, obs
		      if os.path.isfile(path + suite + '/' + dateS + '/DIA/' + dateS + '00_SAM.dia_' + obs):
			  	count[list_obs.index(obs), num_dates,:]     = read_file(path, suite, dateS, obs)['DATA']
				count_ref[list_obs.index(obs), num_dates,:] = read_file(path_ref, suite_ref, dateS, obs)['DATA']
				omp[list_obs.index(obs), num_dates,:]       = read_file(path, suite, dateS, obs)['AVR(MISFIT)']
				omp_ref[list_obs.index(obs), num_dates,:]   = read_file(path_ref, suite_ref, dateS, obs)['AVR(MISFIT)']
				ms[list_obs.index(obs), num_dates,:]        = read_file(path, suite, dateS, obs)['MS(MISFIT)']
				ms_ref[list_obs.index(obs), num_dates,:]    = read_file(path_ref, suite_ref, dateS, obs)['MS(MISFIT)']
			  else:
			  	count[list_obs.index(obs), num_dates,:] = np.nan
				count_ref[list_obs.index(obs), num_dates,:] = np.nan
				omp[list_obs.index(obs), num_dates,:] = np.nan
				omp_ref[list_obs.index(obs), num_dates,:] = np.nan
				ms[list_obs.index(obs), num_dates,:] = np.nan
				ms_ref[list_obs.index(obs), num_dates,:] = np.nan

# compute RMS
rms = np.sqrt(ms)                                                                               
rms_ref = np.sqrt(ms_ref)

# Creating dataframes 
#SST
df_count_SST = pd.DataFrame(count[list_obs.index('GEN_SST'),:,0], index=dates, columns=['nobs_00'])
df_omp_SST   = pd.DataFrame(omp[list_obs.index('GEN_SST'),:,0], index=dates, columns=['omp_00'])
df_rms_SST   = pd.DataFrame(rms[list_obs.index('GEN_SST'),:,0], index=dates, columns=['rms_00'])
df_count_ref_SST = pd.DataFrame(count_ref[list_obs.index('GEN_SST'),:,0], index=dates, columns=['nobs_00'])
df_omp_ref_SST   = pd.DataFrame(omp_ref[list_obs.index('GEN_SST'),:,0], index=dates, columns=['omp_00'])
df_rms_ref_SST   = pd.DataFrame(rms_ref[list_obs.index('GEN_SST'),:,0], index=dates, columns=['rms_00'])

#SST_night
df_count_SST_night = pd.DataFrame(count[list_obs.index('GEN_SST_night'),:,0], index=dates, columns=['nobs_00'])
df_omp_SST_night   = pd.DataFrame(omp[list_obs.index('GEN_SST_night'),:,0], index=dates, columns=['omp_00'])
df_rms_SST_night   = pd.DataFrame(rms[list_obs.index('GEN_SST_night'),:,0], index=dates, columns=['rms_00'])
df_count_ref_SST_night = pd.DataFrame(count_ref[list_obs.index('GEN_SST_night'),:,0], index=dates, columns=['nobs_00'])
df_omp_ref_SST_night   = pd.DataFrame(omp_ref[list_obs.index('GEN_SST_night'),:,0], index=dates, columns=['omp_00'])
df_rms_ref_SST_night   = pd.DataFrame(rms_ref[list_obs.index('GEN_SST_night'),:,0], index=dates, columns=['rms_00'])

#ALTIKA
df_count_ALTIKA = pd.DataFrame(count[list_obs.index('ALTIKA'),:,0], index=dates, columns=['nobs_00'])
df_omp_ALTIKA   = pd.DataFrame(omp[list_obs.index('ALTIKA'),:,0], index=dates, columns=['omp_00'])
df_rms_ALTIKA   = pd.DataFrame(rms[list_obs.index('ALTIKA'),:,0], index=dates, columns=['rms_00'])
df_count_ref_ALTIKA = pd.DataFrame(count_ref[list_obs.index('ALTIKA'),:,0], index=dates, columns=['nobs_00'])
df_omp_ref_ALTIKA   = pd.DataFrame(omp_ref[list_obs.index('ALTIKA'),:,0], index=dates, columns=['omp_00'])
df_rms_ref_ALTIKA   = pd.DataFrame(rms_ref[list_obs.index('ALTIKA'),:,0], index=dates, columns=['rms_00'])

#CRYOSAT2
df_count_CRYOSAT2 = pd.DataFrame(count[list_obs.index('CRYOSAT2'),:,0], index=dates, columns=['nobs_00'])
df_omp_CRYOSAT2   = pd.DataFrame(omp[list_obs.index('CRYOSAT2'),:,0], index=dates, columns=['omp_00'])
df_rms_CRYOSAT2   = pd.DataFrame(rms[list_obs.index('CRYOSAT2'),:,0], index=dates, columns=['rms_00'])
df_count_ref_CRYOSAT2 = pd.DataFrame(count_ref[list_obs.index('CRYOSAT2'),:,0], index=dates, columns=['nobs_00'])
df_omp_ref_CRYOSAT2   = pd.DataFrame(omp_ref[list_obs.index('CRYOSAT2'),:,0], index=dates, columns=['omp_00'])
df_rms_ref_CRYOSAT2   = pd.DataFrame(rms_ref[list_obs.index('CRYOSAT2'),:,0], index=dates, columns=['rms_00'])

#JASON2
df_count_JASON2 = pd.DataFrame(count[list_obs.index('JASON2'),:,0], index=dates, columns=['nobs_00'])
df_omp_JASON2   = pd.DataFrame(omp[list_obs.index('JASON2'),:,0], index=dates, columns=['omp_00'])
df_rms_JASON2   = pd.DataFrame(rms[list_obs.index('JASON2'),:,0], index=dates, columns=['rms_00'])
df_count_ref_JASON2 = pd.DataFrame(count_ref[list_obs.index('JASON2'),:,0], index=dates, columns=['nobs_00'])
df_omp_ref_JASON2   = pd.DataFrame(omp_ref[list_obs.index('JASON2'),:,0], index=dates, columns=['omp_00'])
df_rms_ref_JASON2   = pd.DataFrame(rms_ref[list_obs.index('JASON2'),:,0], index=dates, columns=['rms_00'])

#JASON3
df_count_JASON3 = pd.DataFrame(count[list_obs.index('JASON3'),:,0], index=dates, columns=['nobs_00'])
df_omp_JASON3   = pd.DataFrame(omp[list_obs.index('JASON3'),:,0], index=dates, columns=['omp_00'])
df_rms_JASON3   = pd.DataFrame(rms[list_obs.index('JASON3'),:,0], index=dates, columns=['rms_00'])
df_count_ref_JASON3 = pd.DataFrame(count_ref[list_obs.index('JASON3'),:,0], index=dates, columns=['nobs_00'])
df_omp_ref_JASON3   = pd.DataFrame(omp_ref[list_obs.index('JASON3'),:,0], index=dates, columns=['omp_00'])
df_rms_ref_JASON3   = pd.DataFrame(rms_ref[list_obs.index('JASON3'),:,0], index=dates, columns=['rms_00'])

#JASON2N
df_count_JASON2N = pd.DataFrame(count[list_obs.index('JASON2N'),:,0], index=dates, columns=['nobs_00'])
df_omp_JASON2N   = pd.DataFrame(omp[list_obs.index('JASON2N'),:,0], index=dates, columns=['omp_00'])
df_rms_JASON2N   = pd.DataFrame(rms[list_obs.index('JASON2N'),:,0], index=dates, columns=['rms_00'])
df_count_ref_JASON2N = pd.DataFrame(count_ref[list_obs.index('JASON2N'),:,0], index=dates, columns=['nobs_00'])
df_omp_ref_JASON2N   = pd.DataFrame(omp_ref[list_obs.index('JASON2N'),:,0], index=dates, columns=['omp_00'])
df_rms_ref_JASON2N   = pd.DataFrame(rms_ref[list_obs.index('JASON2N'),:,0], index=dates, columns=['rms_00'])

#HY2A
df_count_HY2A = pd.DataFrame(count[list_obs.index('HY2A'),:,0], index=dates, columns=['nobs_00'])
df_omp_HY2A   = pd.DataFrame(omp[list_obs.index('HY2A'),:,0], index=dates, columns=['omp_00'])
df_rms_HY2A   = pd.DataFrame(rms[list_obs.index('HY2A'),:,0], index=dates, columns=['rms_00'])
df_count_ref_HY2A = pd.DataFrame(count_ref[list_obs.index('HY2A'),:,0], index=dates, columns=['nobs_00'])
df_omp_ref_HY2A   = pd.DataFrame(omp_ref[list_obs.index('HY2A'),:,0], index=dates, columns=['omp_00'])
df_rms_ref_HY2A   = pd.DataFrame(rms_ref[list_obs.index('HY2A'),:,0], index=dates, columns=['rms_00'])

#SENTINEL3A
df_count_S3A = pd.DataFrame(count[list_obs.index('SENTINEL3A'),:,0], index=dates, columns=['nobs_00'])
df_omp_S3A   = pd.DataFrame(omp[list_obs.index('SENTINEL3A'),:,0], index=dates, columns=['omp_00'])
df_rms_S3A   = pd.DataFrame(rms[list_obs.index('SENTINEL3A'),:,0], index=dates, columns=['rms_00'])
df_count_ref_S3A = pd.DataFrame(count_ref[list_obs.index('SENTINEL3A'),:,0], index=dates, columns=['nobs_00'])
df_omp_ref_S3A   = pd.DataFrame(omp_ref[list_obs.index('SENTINEL3A'),:,0], index=dates, columns=['omp_00'])
df_rms_ref_S3A   = pd.DataFrame(rms_ref[list_obs.index('SENTINEL3A'),:,0], index=dates, columns=['rms_00'])

for region in region_num[1:]:
	k = int(region)
	#SST
	df_count_SST['nobs_' + region] = pd.DataFrame(count[list_obs.index('GEN_SST'),:,k], index=dates)
	df_omp_SST['omp_' + region] = pd.DataFrame(omp[list_obs.index('GEN_SST'),:,k], index=dates)
	df_rms_SST['rms_' + region] = pd.DataFrame(rms[list_obs.index('GEN_SST'),:,k], index=dates)
	df_count_ref_SST['nobs_' + region] = pd.DataFrame(count_ref[list_obs.index('GEN_SST'),:,k], index=dates)
	df_omp_ref_SST['omp_' + region] = pd.DataFrame(omp_ref[list_obs.index('GEN_SST'),:,k], index=dates)
	df_rms_ref_SST['rms_' + region] = pd.DataFrame(rms_ref[list_obs.index('GEN_SST'),:,k], index=dates)
	
	#SST_night
	df_count_SST_night['nobs_' + region] = pd.DataFrame(count[list_obs.index('GEN_SST_night'),:,k], index=dates)
	df_omp_SST_night['omp_' + region] = pd.DataFrame(omp[list_obs.index('GEN_SST_night'),:,k], index=dates)
	df_rms_SST_night['rms_' + region] = pd.DataFrame(rms[list_obs.index('GEN_SST_night'),:,k], index=dates)
	df_count_ref_SST_night['nobs_' + region] = pd.DataFrame(count_ref[list_obs.index('GEN_SST_night'),:,k], index=dates)
	df_omp_ref_SST_night['omp_' + region] = pd.DataFrame(omp_ref[list_obs.index('GEN_SST_night'),:,k], index=dates)
	df_rms_ref_SST_night['rms_' + region] = pd.DataFrame(rms_ref[list_obs.index('GEN_SST_night'),:,k], index=dates)

	#ALTIKA
	df_count_ALTIKA['nobs_' + region] = pd.DataFrame(count[list_obs.index('ALTIKA'),:,k], index=dates)
	df_omp_ALTIKA['omp_' + region] = pd.DataFrame(omp[list_obs.index('ALTIKA'),:,k], index=dates)
	df_rms_ALTIKA['rms_' + region] = pd.DataFrame(rms[list_obs.index('ALTIKA'),:,k], index=dates)
	df_count_ref_ALTIKA['nobs_' + region] = pd.DataFrame(count_ref[list_obs.index('ALTIKA'),:,k], index=dates)
	df_omp_ref_ALTIKA['omp_' + region] = pd.DataFrame(omp_ref[list_obs.index('ALTIKA'),:,k], index=dates)
	df_rms_ref_ALTIKA['rms_' + region] = pd.DataFrame(rms_ref[list_obs.index('ALTIKA'),:,k], index=dates)

	#JASON2
	df_count_JASON2['nobs_' + region] = pd.DataFrame(count[list_obs.index('JASON2'),:,k], index=dates)
	df_omp_JASON2['omp_' + region] = pd.DataFrame(omp[list_obs.index('JASON2'),:,k], index=dates)
	df_rms_JASON2['rms_' + region] = pd.DataFrame(rms[list_obs.index('JASON2'),:,k], index=dates)
	df_count_ref_JASON2['nobs_' + region] = pd.DataFrame(count_ref[list_obs.index('JASON2'),:,k], index=dates)
	df_omp_ref_JASON2['omp_' + region] = pd.DataFrame(omp_ref[list_obs.index('JASON2'),:,k], index=dates)
	df_rms_ref_JASON2['rms_' + region] = pd.DataFrame(rms_ref[list_obs.index('JASON2'),:,k], index=dates)

	#JASON3
	df_count_JASON3['nobs_' + region] = pd.DataFrame(count[list_obs.index('JASON3'),:,k], index=dates)
	df_omp_JASON3['omp_' + region] = pd.DataFrame(omp[list_obs.index('JASON3'),:,k], index=dates)
	df_rms_JASON3['rms_' + region] = pd.DataFrame(rms[list_obs.index('JASON3'),:,k], index=dates)
	df_count_ref_JASON3['nobs_' + region] = pd.DataFrame(count_ref[list_obs.index('JASON3'),:,k], index=dates)
	df_omp_ref_JASON3['omp_' + region] = pd.DataFrame(omp_ref[list_obs.index('JASON3'),:,k], index=dates)
	df_rms_ref_JASON3['rms_' + region] = pd.DataFrame(rms_ref[list_obs.index('JASON3'),:,k], index=dates)

	#JASON2N
	df_count_JASON2N['nobs_' + region] = pd.DataFrame(count[list_obs.index('JASON2N'),:,k], index=dates)
	df_omp_JASON2N['omp_' + region] = pd.DataFrame(omp[list_obs.index('JASON2N'),:,k], index=dates)
	df_rms_JASON2N['rms_' + region] = pd.DataFrame(rms[list_obs.index('JASON2N'),:,k], index=dates)
	df_count_ref_JASON2N['nobs_' + region] = pd.DataFrame(count_ref[list_obs.index('JASON2N'),:,k], index=dates)
	df_omp_ref_JASON2N['omp_' + region] = pd.DataFrame(omp_ref[list_obs.index('JASON2N'),:,k], index=dates)
	df_rms_ref_JASON2N['rms_' + region] = pd.DataFrame(rms_ref[list_obs.index('JASON2N'),:,k], index=dates)

	#HY2A
	df_count_HY2A['nobs_' + region] = pd.DataFrame(count[list_obs.index('HY2A'),:,k], index=dates)
	df_omp_HY2A['omp_' + region] = pd.DataFrame(omp[list_obs.index('HY2A'),:,k], index=dates)
	df_rms_HY2A['rms_' + region] = pd.DataFrame(rms[list_obs.index('HY2A'),:,k], index=dates)
	df_count_ref_HY2A['nobs_' + region] = pd.DataFrame(count_ref[list_obs.index('HY2A'),:,k], index=dates)
	df_omp_ref_HY2A['omp_' + region] = pd.DataFrame(omp_ref[list_obs.index('HY2A'),:,k], index=dates)
	df_rms_ref_HY2A['rms_' + region] = pd.DataFrame(rms_ref[list_obs.index('HY2A'),:,k], index=dates)

	#CRYOSAT2
	df_count_CRYOSAT2['nobs_' + region] = pd.DataFrame(count[list_obs.index('CRYOSAT2'),:,k], index=dates)
	df_omp_CRYOSAT2['omp_' + region] = pd.DataFrame(omp[list_obs.index('CRYOSAT2'),:,k], index=dates)
	df_rms_CRYOSAT2['rms_' + region] = pd.DataFrame(rms[list_obs.index('CRYOSAT2'),:,k], index=dates)
	df_count_ref_CRYOSAT2['nobs_' + region] = pd.DataFrame(count_ref[list_obs.index('CRYOSAT2'),:,k], index=dates)
	df_omp_ref_CRYOSAT2['omp_' + region] = pd.DataFrame(omp_ref[list_obs.index('CRYOSAT2'),:,k], index=dates)
	df_rms_ref_CRYOSAT2['rms_' + region] = pd.DataFrame(rms_ref[list_obs.index('CRYOSAT2'),:,k], index=dates)

	#SENTINEL3A
	df_count_S3A['nobs_' + region] = pd.DataFrame(count[list_obs.index('SENTINEL3A'),:,k], index=dates)
	df_omp_S3A['omp_' + region] = pd.DataFrame(omp[list_obs.index('SENTINEL3A'),:,k], index=dates)
	df_rms_S3A['rms_' + region] = pd.DataFrame(rms[list_obs.index('SENTINEL3A'),:,k], index=dates)
	df_count_ref_S3A['nobs_' + region] = pd.DataFrame(count_ref[list_obs.index('SENTINEL3A'),:,k], index=dates)
	df_omp_ref_S3A['omp_' + region] = pd.DataFrame(omp_ref[list_obs.index('SENTINEL3A'),:,k], index=dates)
	df_rms_ref_S3A['rms_' + region] = pd.DataFrame(rms_ref[list_obs.index('SENTINEL3A'),:,k], index=dates)


# Concatenate dataframes
df_SST     = pd.concat([df_count_SST, df_omp_SST, df_rms_SST], axis=1)
df_ref_SST = pd.concat([df_count_ref_SST, df_omp_ref_SST, df_rms_ref_SST], axis=1)
df_SST_night     = pd.concat([df_count_SST_night, df_omp_SST_night, df_rms_SST_night], axis=1)
df_ref_SST_night = pd.concat([df_count_ref_SST_night, df_omp_ref_SST_night, df_rms_ref_SST_night], axis=1)
df_ALTIKA  = pd.concat([df_count_ALTIKA, df_omp_ALTIKA, df_rms_ALTIKA], axis=1)
df_ref_ALTIKA  = pd.concat([df_count_ref_ALTIKA, df_omp_ref_ALTIKA, df_rms_ref_ALTIKA], axis=1)
df_JASON2  = pd.concat([df_count_JASON2, df_omp_JASON2, df_rms_JASON2], axis=1)
df_ref_JASON2  = pd.concat([df_count_ref_JASON2, df_omp_ref_JASON2, df_rms_ref_JASON2], axis=1)
df_JASON3  = pd.concat([df_count_JASON3, df_omp_JASON3, df_rms_JASON3], axis=1)
df_ref_JASON3  = pd.concat([df_count_ref_JASON3, df_omp_ref_JASON3, df_rms_ref_JASON3], axis=1)
df_JASON2N  = pd.concat([df_count_JASON2N, df_omp_JASON2N, df_rms_JASON2N], axis=1)
df_ref_JASON2N  = pd.concat([df_count_ref_JASON2N, df_omp_ref_JASON2N, df_rms_ref_JASON2N], axis=1)
df_CRYOSAT2  = pd.concat([df_count_CRYOSAT2, df_omp_CRYOSAT2, df_rms_CRYOSAT2], axis=1)
df_ref_CRYOSAT2  = pd.concat([df_count_ref_CRYOSAT2, df_omp_ref_CRYOSAT2, df_rms_ref_CRYOSAT2], axis=1)
df_HY2A  = pd.concat([df_count_HY2A, df_omp_HY2A, df_rms_HY2A], axis=1)
df_ref_HY2A  = pd.concat([df_count_ref_HY2A, df_omp_ref_HY2A, df_rms_ref_HY2A], axis=1)
df_S3A  = pd.concat([df_count_S3A, df_omp_S3A, df_rms_S3A], axis=1)
df_ref_S3A  = pd.concat([df_count_ref_S3A, df_omp_ref_S3A, df_rms_ref_S3A], axis=1)



# Filter dates with zero data number and average
df_SST_m = filter_zeros_avg(df_SST)
df_ref_SST_m = filter_zeros_avg(df_ref_SST)
df_SST_night_m = filter_zeros_avg(df_SST_night)
df_ref_SST_night_m = filter_zeros_avg(df_ref_SST_night)
df_ALTIKA_m = filter_zeros_avg(df_ALTIKA)
df_ref_ALTIKA_m = filter_zeros_avg(df_ref_ALTIKA)
df_JASON2_m = filter_zeros_avg(df_JASON2)
df_ref_JASON2_m = filter_zeros_avg(df_ref_JASON2)
df_JASON3_m = filter_zeros_avg(df_JASON3)
df_ref_JASON3_m = filter_zeros_avg(df_ref_JASON3)
df_JASON2N_m = filter_zeros_avg(df_JASON2N)
df_ref_JASON2N_m = filter_zeros_avg(df_ref_JASON2N)
df_CRYOSAT2_m = filter_zeros_avg(df_CRYOSAT2)
df_ref_CRYOSAT2_m = filter_zeros_avg(df_ref_CRYOSAT2)
df_HY2A_m = filter_zeros_avg(df_HY2A)
df_ref_HY2A_m = filter_zeros_avg(df_ref_HY2A)
df_S3A_m = filter_zeros_avg(df_S3A)
df_ref_S3A_m = filter_zeros_avg(df_ref_S3A)

#sys.exit('exit for now')

# Figures production
print 'Figures production ...'
label_cycle= suite
label_ref  = suite_ref

for reg in region_num:
	#SST
	f, [ax1, ax2, ax3] = plt.subplots(3, figsize=(12,10))
	plt.subplots_adjust(hspace=0.3)
  	plt.suptitle('SST diagnostics' , fontsize=14)  
	ax1.plot(df_SST.index, df_SST['nobs_'+reg], color='r', linewidth=1.5, linestyle='-', label=label_cycle)
  	ax1.plot(df_ref_SST.index, df_ref_SST['nobs_'+reg], color='b', linewidth=1.5, linestyle='-', label=label_ref)
	ax1.set_title('NUM(DATA)', fontsize=12, horizontalalignment='left') 
  	ax1.set_ylim([0, df_SST['nobs_'+reg].max() + df_SST['nobs_'+reg].max()/10.])
	ax1.set_xlim(pd.Timestamp(start_date), pd.Timestamp(final_date))
	ax1.yaxis.set_major_formatter(FormatStrFormatter('%d'))
	ax1.text(0.0, 1.02, reg + '-' + regions[int(reg)], size=12, ha="left", transform=ax1.transAxes)
	ax1.legend(fontsize=12)

  	ax2.plot(df_SST.index, df_SST['omp_'+reg], color='r', linewidth=1.5, linestyle='-')
	ax2.plot(df_ref_SST.index, df_ref_SST['omp_'+reg], color='b', linewidth=1.5, linestyle='-')
	ax2.set_title('AVR(MISFIT)', fontsize=12, horizontalalignment='left')
	ax2.set_ylabel('deg', fontsize=12, horizontalalignment='left')
	ax2.axhline(y=0, linewidth=1, color='k', zorder = 0)
	ax2.set_ylim([-1,1])
	ax2.set_xlim(pd.Timestamp(start_date), pd.Timestamp(final_date))
	ax2.set_yticks(np.arange(-1, 1.1, 0.2))
	ax2.yaxis.set_major_locator(MultipleLocator(0.5))
	ax2.yaxis.set_major_formatter(FormatStrFormatter('%1.1f'))
	ax2.yaxis.set_minor_locator(MultipleLocator(0.1))

  	ax3.plot(df_SST.index, df_SST['rms_'+reg], color='r', linewidth=1.5, linestyle='-')
	ax3.plot(df_ref_SST.index, df_ref_SST['rms_'+reg], color='b', linewidth=1.5, linestyle='-')
	ax3.set_title('RMS(MISFIT)', fontsize=12, horizontalalignment='left')
  	ax3.set_ylabel('deg', fontsize=12, horizontalalignment='left')
	ax3.set_ylim([0,2])
	ax3.set_xlim(pd.Timestamp(start_date), pd.Timestamp(final_date))
	ax3.set_yticks(np.arange(0, 2.1, 0.2))
	ax3.yaxis.set_major_locator(MultipleLocator(0.5))
  	ax3.yaxis.set_major_formatter(FormatStrFormatter('%1.1f'))
	ax3.yaxis.set_minor_locator(MultipleLocator(0.1))
   	plt.savefig(output_path + suite + '/GEN_SST/GEN_SST_'+start_date+'_'+final_date+'_'+str(int(reg))+'.png', dpi=80)
	print 'Done for SST ...' + regions[int(reg)]
	plt.close()

	#SST_night
	f, [ax1, ax2, ax3] = plt.subplots(3, figsize=(12,10))
	plt.subplots_adjust(hspace=0.3)
  	plt.suptitle('SST_night diagnostics' , fontsize=14)  
	ax1.plot(df_SST_night.index, df_SST_night['nobs_'+reg], color='r', linewidth=1.5, linestyle='-', label=label_cycle)
  	ax1.plot(df_ref_SST_night.index, df_ref_SST_night['nobs_'+reg], color='b', linewidth=1.5, linestyle='-', label=label_ref)
  	ax1.set_title('NUM(DATA)', fontsize=12, horizontalalignment='left') 
  	ax1.set_ylim([0, df_SST_night['nobs_'+reg].max() + df_SST_night['nobs_'+reg].max()/10.])
	ax1.set_xlim(pd.Timestamp(start_date), pd.Timestamp(final_date))
  	ax1.yaxis.set_major_formatter(FormatStrFormatter('%d'))
	ax1.text(0.0, 1.02, reg + '-' + regions[int(reg)], size=12, ha="left", transform=ax1.transAxes)
    ax1.legend(fontsize=12)

  	ax2.plot(df_SST_night.index, df_SST_night['omp_'+reg], color='r', linewidth=1.5, linestyle='-')
  	ax2.plot(df_ref_SST_night.index, df_ref_SST_night['omp_'+reg], color='b', linewidth=1.5, linestyle='-')
  	ax2.set_title('AVR(MISFIT)', fontsize=12, horizontalalignment='left')
  	ax2.set_ylabel('deg', fontsize=12, horizontalalignment='left')
  	ax2.axhline(y=0, linewidth=1, color='k', zorder = 0)
  	ax2.set_ylim([-1,1])
	ax2.set_xlim(pd.Timestamp(start_date), pd.Timestamp(final_date))
  	ax2.set_yticks(np.arange(-1, 1.1, 0.2))
  	ax2.yaxis.set_major_locator(MultipleLocator(0.5))
  	ax2.yaxis.set_major_formatter(FormatStrFormatter('%1.1f'))
  	ax2.yaxis.set_minor_locator(MultipleLocator(0.1))

  	ax3.plot(df_SST_night.index, df_SST_night['rms_'+reg], color='r', linewidth=1.5, linestyle='-')
  	ax3.plot(df_ref_SST_night.index, df_ref_SST_night['rms_'+reg], color='b', linewidth=1.5, linestyle='-')
  	ax3.set_title('RMS(MISFIT)', fontsize=12, horizontalalignment='left')
  	ax3.set_ylabel('deg', fontsize=12, horizontalalignment='left')
  	ax3.set_ylim([0,2])
	ax3.set_xlim(pd.Timestamp(start_date), pd.Timestamp(final_date))
  	ax3.set_yticks(np.arange(0, 2.1, 0.2))
  	ax3.yaxis.set_major_locator(MultipleLocator(0.5))
  	ax3.yaxis.set_major_formatter(FormatStrFormatter('%1.1f'))
  	ax3.yaxis.set_minor_locator(MultipleLocator(0.1))
  
   	plt.savefig(output_path + suite + '/GEN_SST_night/GEN_SST_night_'+start_date+'_'+final_date+'_'+str(int(reg))+'.png', dpi=80)
	print 'Done for SST_night ...' + regions[int(reg)]
	plt.close()

	#ALTIKA
	f, [ax1, ax2, ax3] = plt.subplots(3, figsize=(12,10))
	plt.subplots_adjust(hspace=0.3)
  	plt.suptitle('SLA-ALTIKA diagnostics' , fontsize=14)  
	ax1.plot(df_ALTIKA.index, df_ALTIKA['nobs_'+reg], color='r', linewidth=1.5, linestyle='-', label=label_cycle)
  	ax1.plot(df_ref_ALTIKA.index, df_ref_ALTIKA['nobs_'+reg], color='b', linewidth=1.5, linestyle='-', label=label_ref)
  	ax1.set_title('NUM(DATA)', fontsize=12, horizontalalignment='left') 
  	ax1.set_ylim([0, df_ALTIKA['nobs_'+reg].max() + df_ALTIKA['nobs_'+reg].max()/10.])
	ax1.set_xlim(pd.Timestamp(start_date), pd.Timestamp(final_date))
  	ax1.yaxis.set_major_formatter(FormatStrFormatter('%d'))
	ax1.text(0.0, 1.02, reg + '-' + regions[int(reg)], size=12, ha="left", transform=ax1.transAxes)
    ax1.legend(fontsize=12)

  	ax2.plot(df_ALTIKA.index, df_ALTIKA['omp_'+reg]*100, color='r', linewidth=1.5, linestyle='-')
  	ax2.plot(df_ref_ALTIKA.index, df_ref_ALTIKA['omp_'+reg]*100, color='b', linewidth=1.5, linestyle='-')
  	ax2.set_title('AVR(MISFIT)', fontsize=12, horizontalalignment='left')
  	ax2.set_ylabel('cm', fontsize=12, horizontalalignment='left')
  	ax2.axhline(y=0, linewidth=1, color='k', zorder = 0)
  	ax2.set_ylim([-6, 6])
	ax2.set_xlim(pd.Timestamp(start_date), pd.Timestamp(final_date))
  	ax2.set_yticks(np.arange(-6, 6.1, 0.5))
  	ax2.yaxis.set_major_locator(MultipleLocator(2.0))
  	ax2.yaxis.set_major_formatter(FormatStrFormatter('%d'))
  	ax2.yaxis.set_minor_locator(MultipleLocator(0.5))

  	ax3.plot(df_ALTIKA.index, df_ALTIKA['rms_'+reg]*100, color='r', linewidth=1.5, linestyle='-')
  	ax3.plot(df_ref_ALTIKA.index, df_ref_ALTIKA['rms_'+reg]*100, color='b', linewidth=1.5, linestyle='-')
  	ax3.set_title('RMS(MISFIT)', fontsize=12, horizontalalignment='left')
  	ax3.set_ylabel('cm', fontsize=12, horizontalalignment='left')
  	ax3.set_ylim([0, 30])
	ax3.set_xlim(pd.Timestamp(start_date), pd.Timestamp(final_date))
  	ax3.set_yticks(np.arange(0, 30.1, 1))
  	ax3.yaxis.set_major_locator(MultipleLocator(5.0))
  	ax3.yaxis.set_major_formatter(FormatStrFormatter('%d'))
  	ax3.yaxis.set_minor_locator(MultipleLocator(1.0))
  
   	plt.savefig(output_path + suite + '/ALTIKA/ALTIKA_'+start_date+'_'+final_date+'_'+str(int(reg))+'.png', dpi=80)
	print 'Done for ALTIKA ...' + regions[int(reg)]
	plt.close()

	#JASON2
	f, [ax1, ax2, ax3] = plt.subplots(3, figsize=(12,10))
	plt.subplots_adjust(hspace=0.3)
  	plt.suptitle('SLA-JASON2 diagnostics' , fontsize=14)  
	ax1.plot(df_JASON2.index, df_JASON2['nobs_'+reg], color='r', linewidth=1.5, linestyle='-', label=label_cycle)
  	ax1.plot(df_ref_JASON2.index, df_ref_JASON2['nobs_'+reg], color='b', linewidth=1.5, linestyle='-', label=label_ref)
  	ax1.set_title('NUM(DATA)', fontsize=12, horizontalalignment='left') 
  	ax1.set_ylim([0, df_JASON2['nobs_'+reg].max() + df_JASON2['nobs_'+reg].max()/10.])
	ax1.set_xlim(pd.Timestamp(start_date), pd.Timestamp(final_date))
  	ax1.yaxis.set_major_formatter(FormatStrFormatter('%d'))
	ax1.text(0.0, 1.02, reg + '-' + regions[int(reg)], size=12, ha="left", transform=ax1.transAxes)
    ax1.legend(fontsize=12)

	ax2.plot(df_JASON2.index, df_JASON2['omp_'+reg]*100, color='r', linewidth=1.5, linestyle='-')
  	ax2.plot(df_ref_JASON2.index, df_ref_JASON2['omp_'+reg]*100, color='b', linewidth=1.5, linestyle='-')
  	ax2.set_title('AVR(MISFIT)', fontsize=12, horizontalalignment='left')
  	ax2.set_ylabel('deg', fontsize=10, horizontalalignment='left')
  	ax2.axhline(y=0, linewidth=1, color='k', zorder = 0)
  	ax2.set_ylim([-6,6])
	ax2.set_xlim(pd.Timestamp(start_date), pd.Timestamp(final_date))
  	ax2.set_yticks(np.arange(-6, 6.1, 0.5))
  	ax2.yaxis.set_major_locator(MultipleLocator(2.0))
  	ax2.yaxis.set_major_formatter(FormatStrFormatter('%d'))
  	ax2.yaxis.set_minor_locator(MultipleLocator(1.0))

  	ax3.plot(df_JASON2.index, df_JASON2['rms_'+reg]*100, color='r', linewidth=1.5, linestyle='-')
  	ax3.plot(df_ref_JASON2.index, df_ref_JASON2['rms_'+reg]*100, color='b', linewidth=1.5, linestyle='-')
  	ax3.set_title('RMS(MISFIT)', fontsize=12, horizontalalignment='left')
  	ax3.set_ylabel('deg', fontsize=10, horizontalalignment='left')
  	ax3.set_ylim([0, 30])
	ax3.set_xlim(pd.Timestamp(start_date), pd.Timestamp(final_date))
  	ax3.set_yticks(np.arange(0, 30.1, 1))
  	ax3.yaxis.set_major_locator(MultipleLocator(5.0))
  	ax3.yaxis.set_major_formatter(FormatStrFormatter('%d'))
  	ax3.yaxis.set_minor_locator(MultipleLocator(1.0))

   	plt.savefig(output_path + suite + '/JASON2/JASON2_'+start_date+'_'+final_date+'_'+str(int(reg))+'.png', dpi=80)
	print 'Done for JASON2 ...' + regions[int(reg)]
	plt.close()

	#JASON3
	f, [ax1, ax2, ax3] = plt.subplots(3, figsize=(12,10))
	plt.subplots_adjust(hspace=0.3)
  	plt.suptitle('SLA-JASON3 diagnostics' , fontsize=14)  
	ax1.plot(df_JASON3.index, df_JASON3['nobs_'+reg], color='r', linewidth=1.5, linestyle='-', label=label_cycle)
  	ax1.plot(df_ref_JASON3.index, df_ref_JASON3['nobs_'+reg], color='b', linewidth=1.5, linestyle='-', label=label_ref)
  	ax1.set_title('NUM(DATA)', fontsize=12, horizontalalignment='left') 
  	ax1.set_ylim([0, df_JASON3['nobs_'+reg].max() + df_JASON3['nobs_'+reg].max()/10.])
	ax1.set_xlim(pd.Timestamp(start_date), pd.Timestamp(final_date))
  	ax1.yaxis.set_major_formatter(FormatStrFormatter('%d'))
	ax1.text(0.0, 1.02, reg + '-' + regions[int(reg)], size=12, ha="left", transform=ax1.transAxes)
    ax1.legend(fontsize=12)

  	ax2.plot(df_JASON3.index, df_JASON3['omp_'+reg]*100, color='r', linewidth=1.5, linestyle='-')
  	ax2.plot(df_ref_JASON3.index, df_ref_JASON3['omp_'+reg]*100, color='b', linewidth=1.5, linestyle='-')
  	ax2.set_title('AVR(MISFIT)', fontsize=12, horizontalalignment='left')
  	ax2.set_ylabel('deg', fontsize=10, horizontalalignment='left')
  	ax2.axhline(y=0, linewidth=1, color='k', zorder = 0)
  	ax2.set_ylim([-6,6])
	ax2.set_xlim(pd.Timestamp(start_date), pd.Timestamp(final_date))
  	ax2.set_yticks(np.arange(-6, 6.1, 0.5))
  	ax2.yaxis.set_major_locator(MultipleLocator(2.0))
  	ax2.yaxis.set_major_formatter(FormatStrFormatter('%d'))
  	ax2.yaxis.set_minor_locator(MultipleLocator(1.0))

  	ax3.plot(df_JASON3.index, df_JASON3['rms_'+reg]*100, color='r', linewidth=1.5, linestyle='-')
  	ax3.plot(df_ref_JASON3.index, df_ref_JASON3['rms_'+reg]*100, color='b', linewidth=1.5, linestyle='-')
  	ax3.set_title('RMS(MISFIT)', fontsize=12, horizontalalignment='left')
  	ax3.set_ylabel('deg', fontsize=10, horizontalalignment='left')
  	ax3.set_ylim([0, 30])
	ax3.set_xlim(pd.Timestamp(start_date), pd.Timestamp(final_date))
  	ax3.set_yticks(np.arange(0, 30.1, 1))
  	ax3.yaxis.set_major_locator(MultipleLocator(5.0))
  	ax3.yaxis.set_major_formatter(FormatStrFormatter('%d'))
  	ax3.yaxis.set_minor_locator(MultipleLocator(1.0))

   	plt.savefig(output_path + suite + '/JASON3/JASON3_'+start_date+'_'+final_date+'_'+str(int(reg))+'.png', dpi=80)
	print 'Done for JASON3 ...' + regions[int(reg)]
	plt.close()

	#JASON2N
	f, [ax1, ax2, ax3] = plt.subplots(3, figsize=(12,10))
	plt.subplots_adjust(hspace=0.3)
  	plt.suptitle('SLA-JASON2N diagnostics' , fontsize=14)  
	ax1.plot(df_JASON2N.index, df_JASON2N['nobs_'+reg], color='r', linewidth=1.5, linestyle='-', label=label_cycle)
  	ax1.plot(df_ref_JASON2N.index, df_ref_JASON2N['nobs_'+reg], color='b', linewidth=1.5, linestyle='-', label=label_ref)
  	ax1.set_title('NUM(DATA)', fontsize=12, horizontalalignment='left') 
  	ax1.set_ylim([0, df_JASON2N['nobs_'+reg].max() + df_JASON2N['nobs_'+reg].max()/10.])
	ax1.set_xlim(pd.Timestamp(start_date), pd.Timestamp(final_date))
  	ax1.yaxis.set_major_formatter(FormatStrFormatter('%d'))
	ax1.text(0.0, 1.02, reg + '-' + regions[int(reg)], size=12, ha="left", transform=ax1.transAxes)
    ax1.legend(fontsize=12)

  	ax2.plot(df_JASON2N.index, df_JASON2N['omp_'+reg]*100, color='r', linewidth=1.5, linestyle='-')
  	ax2.plot(df_ref_JASON2N.index, df_ref_JASON2N['omp_'+reg]*100, color='b', linewidth=1.5, linestyle='-')
  	ax2.set_title('AVR(MISFIT)', fontsize=12, horizontalalignment='left')
  	ax2.set_ylabel('deg', fontsize=10, horizontalalignment='left')
  	ax2.axhline(y=0, linewidth=1, color='k', zorder = 0)
  	ax2.set_ylim([-6,6])
	ax2.set_xlim(pd.Timestamp(start_date), pd.Timestamp(final_date))
  	ax2.set_yticks(np.arange(-6, 6.1, 0.5))
  	ax2.yaxis.set_major_locator(MultipleLocator(2.0))
  	ax2.yaxis.set_major_formatter(FormatStrFormatter('%d'))
  	ax2.yaxis.set_minor_locator(MultipleLocator(1.0))
  	
	ax3.plot(df_JASON2N.index, df_JASON2N['rms_'+reg]*100, color='r', linewidth=1.5, linestyle='-')
  	ax3.plot(df_ref_JASON2N.index, df_ref_JASON2N['rms_'+reg]*100, color='b', linewidth=1.5, linestyle='-')
  	ax3.set_title('RMS(MISFIT)', fontsize=12, horizontalalignment='left')
  	ax3.set_ylabel('deg', fontsize=10, horizontalalignment='left')
  	ax3.set_ylim([0, 30])
	ax3.set_xlim(pd.Timestamp(start_date), pd.Timestamp(final_date))
  	ax3.set_yticks(np.arange(0, 30.1, 1))
  	ax3.yaxis.set_major_locator(MultipleLocator(5.0))
  	ax3.yaxis.set_major_formatter(FormatStrFormatter('%d'))
  	ax3.yaxis.set_minor_locator(MultipleLocator(1.0))

   	plt.savefig(output_path + suite + '/JASON2N/JASON2N_'+start_date+'_'+final_date+'_'+str(int(reg))+'.png', dpi=80)
	print 'Done for JASON2N ...' + regions[int(reg)]
	plt.close()

	#CRYOSAT2
	f, [ax1, ax2, ax3] = plt.subplots(3, figsize=(12,10))
	plt.subplots_adjust(hspace=0.3)
  	plt.suptitle('SLA-CRYOSAT2 diagnostics' , fontsize=14)  
	ax1.plot(df_CRYOSAT2.index, df_CRYOSAT2['nobs_'+reg], color='r', linewidth=1.5, linestyle='-', label=label_cycle)
  	ax1.plot(df_ref_CRYOSAT2.index, df_ref_CRYOSAT2['nobs_'+reg], color='b', linewidth=1.5, linestyle='-', label=label_ref)
  	ax1.set_title('NUM(DATA)', fontsize=12, horizontalalignment='left') 
  	ax1.set_ylim([0, df_CRYOSAT2['nobs_'+reg].max() + df_CRYOSAT2['nobs_'+reg].max()/10.])
	ax1.set_xlim(pd.Timestamp(start_date), pd.Timestamp(final_date))
  	ax1.yaxis.set_major_formatter(FormatStrFormatter('%d'))
	ax1.text(0.0, 1.02, reg + '-' + regions[int(reg)], size=12, ha="left", transform=ax1.transAxes)
    ax1.legend(fontsize=12)

  	ax2.plot(df_CRYOSAT2.index, df_CRYOSAT2['omp_'+reg]*100, color='r', linewidth=1.5, linestyle='-')
  	ax2.plot(df_ref_CRYOSAT2.index, df_ref_CRYOSAT2['omp_'+reg]*100, color='b', linewidth=1.5, linestyle='-')
  	ax2.set_title('AVR(MISFIT)', fontsize=12, horizontalalignment='left')
  	ax2.set_ylabel('deg', fontsize=10, horizontalalignment='left')
  	ax2.axhline(y=0, linewidth=1, color='k', zorder = 0)
  	ax2.set_ylim([-6,6])
	ax2.set_xlim(pd.Timestamp(start_date), pd.Timestamp(final_date))
  	ax2.set_yticks(np.arange(-6, 6.1, 0.5))
  	ax2.yaxis.set_major_locator(MultipleLocator(2.0))
  	ax2.yaxis.set_major_formatter(FormatStrFormatter('%d'))
  	ax2.yaxis.set_minor_locator(MultipleLocator(1.0))

  	ax3.plot(df_CRYOSAT2.index, df_CRYOSAT2['rms_'+reg]*100, color='r', linewidth=1.5, linestyle='-')
  	ax3.plot(df_ref_CRYOSAT2.index, df_ref_CRYOSAT2['rms_'+reg]*100, color='b', linewidth=1.5, linestyle='-')
  	ax3.set_title('RMS(MISFIT)', fontsize=12, horizontalalignment='left')
  	ax3.set_ylabel('deg', fontsize=10, horizontalalignment='left')
  	ax3.set_ylim([0, 30])
	ax3.set_xlim(pd.Timestamp(start_date), pd.Timestamp(final_date))
  	ax3.set_yticks(np.arange(0, 30.1, 1))
  	ax3.yaxis.set_major_locator(MultipleLocator(5.0))
  	ax3.yaxis.set_major_formatter(FormatStrFormatter('%d'))
  	ax3.yaxis.set_minor_locator(MultipleLocator(1.0))

   	plt.savefig(output_path + suite + '/CRYOSAT2/CRYOSAT2_'+start_date+'_'+final_date+'_'+str(int(reg))+'.png', dpi=80)
	print 'Done for CRYOSAT2 ...' + regions[int(reg)]
	plt.close()

	#HY2A
	f, [ax1, ax2, ax3] = plt.subplots(3, figsize=(12,10))
	plt.subplots_adjust(hspace=0.3)
  	plt.suptitle('SLA-HY2A diagnostics' , fontsize=14)  
	ax1.plot(df_HY2A.index, df_HY2A['nobs_'+reg], color='r', linewidth=1.5, linestyle='-', label=label_cycle)
  	ax1.plot(df_ref_HY2A.index, df_ref_HY2A['nobs_'+reg], color='b', linewidth=1.5, linestyle='-', label=label_ref)
  	ax1.set_title('NUM(DATA)', fontsize=12, horizontalalignment='left') 
  	ax1.set_ylim([0, df_HY2A['nobs_'+reg].max() + df_HY2A['nobs_'+reg].max()/10.])
	ax1.set_xlim(pd.Timestamp(start_date), pd.Timestamp(final_date))
  	ax1.yaxis.set_major_formatter(FormatStrFormatter('%d'))
	ax1.text(0.0, 1.02, reg + '-' + regions[int(reg)], size=12, ha="left", transform=ax1.transAxes)
    ax1.legend(fontsize=12)

  	ax2.plot(df_HY2A.index, df_HY2A['omp_'+reg]*100, color='r', linewidth=1.5, linestyle='-')
  	ax2.plot(df_ref_HY2A.index, df_ref_HY2A['omp_'+reg]*100, color='b', linewidth=1.5, linestyle='-')
  	ax2.set_title('AVR(MISFIT)', fontsize=12, horizontalalignment='left')
  	ax2.set_ylabel('deg', fontsize=10, horizontalalignment='left')
  	ax2.axhline(y=0, linewidth=1, color='k', zorder = 0)
  	ax2.set_ylim([-6,6])
	ax2.set_xlim(pd.Timestamp(start_date), pd.Timestamp(final_date))
  	ax2.set_yticks(np.arange(-6, 6.1, 0.5))
  	ax2.yaxis.set_major_locator(MultipleLocator(2.0))
  	ax2.yaxis.set_major_formatter(FormatStrFormatter('%d'))
  	ax2.yaxis.set_minor_locator(MultipleLocator(1.0))
  	
	ax3.plot(df_HY2A.index, df_HY2A['rms_'+reg]*100, color='r', linewidth=1.5, linestyle='-')
  	ax3.plot(df_ref_HY2A.index, df_ref_HY2A['rms_'+reg]*100				, color='b', linewidth=1.5, linestyle='-')
  	ax3.set_title('RMS(MISFIT)', fontsize=12, horizontalalignment='left')
  	ax3.set_ylabel('deg', fontsize=10, horizontalalignment='left')
  	ax3.set_ylim([0, 30])
	ax3.set_xlim(pd.Timestamp(start_date), pd.Timestamp(final_date))
  	ax3.set_yticks(np.arange(0, 30.1, 1))
  	ax3.yaxis.set_major_locator(MultipleLocator(5.0))
  	ax3.yaxis.set_major_formatter(FormatStrFormatter('%d'))
  	ax3.yaxis.set_minor_locator(MultipleLocator(1.0))
   	
   	plt.savefig(output_path + suite + '/HY2A/HY2A_'+start_date+'_'+final_date+'_'+str(int(reg))+'.png', dpi=80)
	print 'Done for HY2A ...' + regions[int(reg)]
	plt.close()

	#SENTINEL3A
	f, [ax1, ax2, ax3] = plt.subplots(3, figsize=(12,10))
	plt.subplots_adjust(hspace=0.3)
  	plt.suptitle('SLA-SENTINEL3A diagnostics' , fontsize=14)  
	ax1.plot(df_S3A.index, df_S3A['nobs_'+reg], color='r', linewidth=1.5, linestyle='-', label=label_cycle)
  	ax1.plot(df_ref_S3A.index, df_ref_S3A['nobs_'+reg], color='b', linewidth=1.5, linestyle='-', label=label_ref)
  	ax1.set_title('NUM(DATA)', fontsize=12, horizontalalignment='left') 
  	ax1.set_ylim([0, df_S3A['nobs_'+reg].max() + df_S3A['nobs_'+reg].max()/10.])
	ax1.set_xlim(pd.Timestamp(start_date), pd.Timestamp(final_date))
  	ax1.yaxis.set_major_formatter(FormatStrFormatter('%d'))
	ax1.text(0.0, 1.02, reg + '-' + regions[int(reg)], size=12, ha="left", transform=ax1.transAxes)
    ax1.legend(fontsize=12)

  	ax2.plot(df_S3A.index, df_S3A['omp_'+reg]*100, color='r', linewidth=1.5, linestyle='-')
  	ax2.plot(df_ref_S3A.index, df_ref_S3A['omp_'+reg]*100, color='b', linewidth=1.5, linestyle='-')
  	ax2.set_title('AVR(MISFIT)', fontsize=12, horizontalalignment='left')
  	ax2.set_ylabel('deg', fontsize=10, horizontalalignment='left')
  	ax2.axhline(y=0, linewidth=1, color='k', zorder = 0)
  	ax2.set_ylim([-6,6])
  	ax2.set_yticks(np.arange(-6, 6.1, 0.5))
	ax2.set_xlim(pd.Timestamp(start_date), pd.Timestamp(final_date))
  	ax2.yaxis.set_major_locator(MultipleLocator(2.0))
  	ax2.yaxis.set_major_formatter(FormatStrFormatter('%d'))
  	ax2.yaxis.set_minor_locator(MultipleLocator(1.0))
  	
	ax3.plot(df_S3A.index, df_S3A['rms_'+reg]*100, color='r', linewidth=1.5, linestyle='-')
  	ax3.plot(df_ref_S3A.index, df_ref_S3A['rms_'+reg]*100, color='b', linewidth=1.5, linestyle='-')
  	ax3.set_title('RMS(MISFIT)', fontsize=12, horizontalalignment='left')
  	ax3.set_ylabel('deg', fontsize=10, horizontalalignment='left')
  	ax3.set_ylim([0, 30])
  	ax3.set_yticks(np.arange(0, 30.1, 1))
	ax3.set_xlim(pd.Timestamp(start_date), pd.Timestamp(final_date))
  	ax3.yaxis.set_major_locator(MultipleLocator(5.0))
  	ax3.yaxis.set_major_formatter(FormatStrFormatter('%d'))
  	ax3.yaxis.set_minor_locator(MultipleLocator(1.0))
   	
   	plt.savefig(output_path + suite + '/SENTINEL3A/SENTINEL3A_'+start_date+'_'+final_date+'_'+str(int(reg))+'.png', dpi=80)
	print 'Done for SENTINEL3A ...' + regions[int(reg)]
	plt.close()

# Student t-test (only for RMS)
t_SST = np.zeros(47); p_SST = np.zeros(47); 
t_SST_night = np.zeros(47); p_SST_night = np.zeros(47)
t_ALTIKA = np.zeros(47); p_ALTIKA = np.zeros(47)
t_JASON2 = np.zeros(47); p_JASON2 = np.zeros(47)
t_JASON3 = np.zeros(47); p_JASON3 = np.zeros(47)
t_JASON2N = np.zeros(47); p_JASON2N = np.zeros(47)
t_CRYOSAT2 = np.zeros(47); p_CRYOSAT2 = np.zeros(47); 
t_HY2A = np.zeros(47); p_HY2A = np.zeros(47) 
t_S3A = np.zeros(47); p_S3A = np.zeros(47) 


for i in range(94, len(df_SST.columns)):
	t_SST[i-94], p_SST[i-94] = stats.ttest_ind(df_SST.values[:, i], df_ref_SST.values[:,i])
	t_SST_night[i-94], p_SST_night[i-94] = stats.ttest_ind(df_SST_night.values[:, i], df_ref_SST_night.values[:,i])
	t_ALTIKA[i-94], p_ALTIKA[i-94] = stats.ttest_ind(df_ALTIKA.values[:, i], df_ref_ALTIKA.values[:,i])
	t_JASON2[i-94], p_JASON2[i-94] = stats.ttest_ind(df_JASON2.values[:, i], df_ref_JASON2.values[:,i])
	t_JASON3[i-94], p_JASON3[i-94] = stats.ttest_ind(df_JASON3.values[:, i], df_ref_JASON3.values[:,i])
	t_JASON2N[i-94], p_JASON2N[i-94] = stats.ttest_ind(df_JASON2N.values[:, i], df_ref_JASON2N.values[:,i])
	t_CRYOSAT2[i-94], p_CRYOSAT2[i-94] = stats.ttest_ind(df_CRYOSAT2.values[:, i], df_ref_CRYOSAT2.values[:,i])
	t_HY2A[i-94], p_HY2A[i-94] = stats.ttest_ind(df_HY2A.values[:, i], df_ref_HY2A.values[:,i])
	t_S3A[i-94], p_S3A[i-94] = stats.ttest_ind(df_S3A.values[:, i], df_ref_S3A.values[:,i])


# Mean values figures (horizontal bars figures)

plot_mean_stat(df_SST_m, df_ref_SST_m, 'GEN_SST', suite, suite_ref, p_SST)
plot_mean_stat(df_SST_night_m, df_ref_SST_night_m, 'GEN_SST_night', suite, suite_ref, p_SST_night)
plot_mean_stat(df_ALTIKA_m, df_ref_ALTIKA_m, 'ALTIKA', suite, suite_ref, p_ALTIKA)
plot_mean_stat(df_JASON2_m, df_ref_JASON2_m, 'JASON2', suite, suite_ref, p_JASON2)
plot_mean_stat(df_JASON3_m, df_ref_JASON3_m, 'JASON3', suite, suite_ref, p_JASON3)
plot_mean_stat(df_JASON2N_m, df_ref_JASON2N_m, 'JASON2N', suite, suite_ref, p_JASON2N)
plot_mean_stat(df_CRYOSAT2_m, df_ref_CRYOSAT2_m, 'CRYOSAT2', suite, suite_ref, p_CRYOSAT2)
plot_mean_stat(df_HY2A_m, df_ref_HY2A_m, 'HY2A', suite, suite_ref, p_HY2A)
plot_mean_stat(df_S3A_m, df_ref_S3A_m, 'SENTINEL3A', suite, suite_ref, p_S3A)


print 'Finished in ... ', time.time() - initial_time, ' secondes'
