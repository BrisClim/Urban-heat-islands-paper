from __future__ import division
import math
import glob
import iris
import iris.coord_categorisation as icc
import datetime
import simplejson
import numpy as np
import numpy.ma as ma
import statsmodels.api as sm
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, FixedLocator, FormatStrFormatter
from transverse_surface_types import box_top_bottom_grids
from correct_orog_to_ref import temp_ref_height


''' This script plots seasonal UHI '''
''' in regions in England '''
''' urban/rural defined by surface types '''
''' with options to bias correct JJA means '''
''' Eunice Lo '''
''' 05/08/2019 '''
''' updated 03/04/2020 '''


if __name__ == "__main__":
    
    # path settings
    ukcp_dir = "/export/anthropocene/array-01/yl17544/ukcp18_data/12km/"
    out_dir = "/home/bridge/yl17544/UHI_results/"

    # options
    season = "JJA"
    regs = ["London", "Manchester", "Birmingham", "Leeds", "Liverpool", "Southampton", "Newcastle", "Nottingham", "Sheffield", "Bristol"]
    ngrids = 1
    centre_latlons = [[51.5074,0.1278], [53.4808,-2.2426], [52.4862,-1.8904], [53.8008,-1.5491], [53.4084,-2.9916], [50.9097,-1.4044], [54.9783,-1.6178], [52.9548,-1.1581], [53.3811,-1.4701], [51.4545,-2.5879]]
    syr = 1981
    eyr = 2079
    bias_corr = True    # only the mean

    # files
    ukcp_tmax_files = glob.glob(ukcp_dir+"tasmax/day/run*/tasmax_rcp85_land-rcm_uk_12km_*_day_19801201-20801130.nc") 
    ukcp_tmin_files = glob.glob(ukcp_dir+"tasmin/day/run*/tasmin_rcp85_land-rcm_uk_12km_*_day_19801201-20801130.nc")

    # load data
    print("Loading tasmax & tasmin data...")
    years = iris.Constraint(time=lambda cell: syr <= cell.point.year <= eyr) 
    ukcp_tmax_cubes = iris.load(ukcp_tmax_files, "air_temperature" & years)
    ukcp_tmin_cubes = iris.load(ukcp_tmin_files, "air_temperature" & years)

    if bias_corr:
        print("Loading bias correction factors...")
        bc_tmax_arr = np.load("/home/bridge/yl17544/intermediate_output/from_compute_UKCP18_hadukgrid_biascorr_factors/hadukgrid_ukcp18raw_12mems_top10_5by5_2grids_1981-2017mean_JJA_tasmax_biascorrection_factors.npy")
        bc_tmin_arr = np.load("/home/bridge/yl17544/intermediate_output/from_compute_UKCP18_hadukgrid_biascorr_factors/hadukgrid_ukcp18raw_12mems_top10_5by5_2grids_1981-2017mean_JJA_tasmin_biascorrection_factors.npy")
     
        
    print("# grids = ", str(ngrids))
    
    # plot settings
    font = {'size'   : 24}
    plt.rc('font', **font)
    mpl.rcParams['xtick.major.pad'] = 5
    mpl.rcParams['xtick.major.pad'] = 5
    mpl.rcParams['ytick.major.pad'] = 5     
    mpl.rcParams['hatch.color'] = "white"
     
    f, axarr = plt.subplots(5, 2, figsize=(18,12), sharex='col', sharey='row', num=1)
    spx = np.tile(np.array([0, 1]), 5)
    spy = np.repeat(np.arange(0, 5), 2)
    p = 0
    
    # for JJA
    season_cons = iris.Constraint(time=lambda cell: 6 <= cell.point.month <= 8)
    ukcp_tmax_jja = ukcp_tmax_cubes.extract(season_cons)
    ukcp_tmin_jja = ukcp_tmin_cubes.extract(season_cons)
    tmax_trd = np.zeros((len(regs)))
    tmax_trd_pvals = np.zeros((len(regs)))
    tmax_tall = np.zeros((len(regs), 12))
    tmax_tlh = np.zeros((len(regs), 2))
 
    tmin_trd = np.zeros((len(regs)))
    tmin_trd_pvals = np.zeros((len(regs)))
    tmin_tall = np.zeros((len(regs), 12))
    tmin_tlh = np.zeros((len(regs), 2))
    '''
    # for hottest 3 days
    tmax_hw_trd = np.zeros((len(regs)))
    tmax_hw_trd_pvals = np.zeros((len(regs)))
    tmax_hw_tall = np.zeros((len(regs), 12))
    tmax_hw_tlh = np.zeros((len(regs), 2))    

    tmin_hw_trd = np.zeros((len(regs)))
    tmin_hw_trd_pvals = np.zeros((len(regs))) 
    tmin_hw_tall = np.zeros((len(regs), 12))
    tmin_hw_tlh = np.zeros((len(regs), 2))
    ''' 
    # array indicating whether to exclude the region or not
    excl_regs = np.zeros((len(regs)), dtype=bool)    # True = exclude | False = include; default False

    for n in range(len(regs)):

        print("---------------------------")
        print(regs[n])
        print("Finding urban/rural grids")
        # Urban/rural mask, static as urban vs non-urban do not change over time, checked
        tfracs, tinds, bfracs, binds, yxinds = box_top_bottom_grids(2017, 2018, ngrids, centre_latlons[n])
        tfracs = tfracs[-1]     # 2018 only
        tinds = tinds[-1]       # 2018 only
        bfracs = bfracs[-1]     # 2018 only
        binds = binds[-1]       # 2018 only 

        print("Top urban fracs: "+str(tfracs))
        print("Bottom urban fracs: "+str(bfracs))

        # Check whether tfracs are all > 0.1?
        excl = np.any(tfracs <= 0.1)   
        excl_regs[n] = excl
        # if excl=True, do not plot!
        if excl:
            print("Some grids <= 0.1, excluding region...")
            axarr[spy[p], spx[p]].set_title(regs[n])
        else:
            print("Computing annual & heatwaves UHIs & trends...")  
            # for JJA
            tmax_UHI = iris.cube.CubeList()
            tmin_UHI = iris.cube.CubeList()  
            '''     
            # for hottest 3 days UHIs, 1981-2079 (12 members x 99 years)
            tmax_hw_UHI = np.loadtxt("/home/bridge/yl17544/UHI_results/heatwave_startends/ukcp18member_annual_hottest3days_1981-2079_"+regs[n]+"_5by5_tasmax_UHI_"+str(ngrids)+"grids.txt").reshape(12, 99)
            tmin_hw_UHI = np.loadtxt("/home/bridge/yl17544/UHI_results/heatwave_startends/ukcp18member_annual_hottest3days_1981-2079_"+regs[n]+"_5by5_tasmin_UHI_"+str(ngrids)+"grids.txt").reshape(12, 99)
            '''
            for m in range(12):
                emem = int(ukcp_tmax_jja[m].coord("ensemble_member").points[0])
                print("Ensemble member ", str(emem))

                # JJA tasmax
                tmax_cube = ukcp_tmax_jja[m]
                # remove lat and lon coords if exist
                coords = [coord.name() for coord in tmax_cube.coords()]
                if 'latitude' in coords:
                    tmax_cube.remove_coord('latitude')
                if 'longitude' in coords:
                    tmax_cube.remove_coord('longitude')
                # JJA tasmin
                tmin_cube = ukcp_tmin_jja[m]

                # top and bottom 1 grid
                if bias_corr: 
                    tmax_urb = tmax_cube[:, 0, tinds[0][0], tinds[0][1]] + bc_tmax_arr[m, tinds[0][0], tinds[0][1]]
                    tmax_rul = tmax_cube[:, 0, binds[0][0], binds[0][1]] + bc_tmax_arr[m, binds[0][0], binds[0][1]]
                    tmin_urb = tmin_cube[:, 0, tinds[0][0], tinds[0][1]] + bc_tmin_arr[m, tinds[0][0], tinds[0][1]]
                    tmin_rul = tmin_cube[:, 0, binds[0][0], binds[0][1]] + bc_tmin_arr[m, binds[0][0], binds[0][1]]                 
                else:
                    tmax_urb = tmax_cube[:, 0, tinds[0][0], tinds[0][1]]
                    tmax_rul = tmax_cube[:, 0, binds[0][0], binds[0][1]]
                    tmin_urb = tmin_cube[:, 0, tinds[0][0], tinds[0][1]]
                    tmin_rul = tmin_cube[:, 0, binds[0][0], binds[0][1]] 
               
                # average over top and bottom n grids
                for g in range(1, ngrids):
                    if bias_corr:
                        tmax_urb += tmax_cube[:, 0, tinds[g][0], tinds[g][1]] + bc_tmax_arr[m, tinds[g][0], tinds[g][1]]
                        tmax_rul += tmax_cube[:, 0, binds[g][0], binds[g][1]] + bc_tmax_arr[m, binds[g][0], binds[g][1]]
                        tmin_urb += tmin_cube[:, 0, tinds[g][0], tinds[g][1]] + bc_tmin_arr[m, tinds[g][0], tinds[g][1]]
                        tmin_rul += tmin_cube[:, 0, binds[g][0], binds[g][1]] + bc_tmin_arr[m, binds[g][0], binds[g][1]]
                    else:
                        tmax_urb += tmax_cube[:, 0, tinds[g][0], tinds[g][1]]
                        tmax_rul += tmax_cube[:, 0, binds[g][0], binds[g][1]]
                        tmin_urb += tmin_cube[:, 0, tinds[g][0], tinds[g][1]]
                        tmin_rul += tmin_cube[:, 0, binds[g][0], binds[g][1]]
                                      
                ave_tmax_UHI = tmax_urb/ngrids - tmax_rul/ngrids
                icc.add_year(ave_tmax_UHI, "time", name="year")
                tmax_UHI.append(ave_tmax_UHI.aggregated_by("year", iris.analysis.MEAN))
                ave_tmin_UHI = tmin_urb/ngrids - tmin_rul/ngrids
                icc.add_year(ave_tmin_UHI, "time", name="year")
                tmin_UHI.append(ave_tmin_UHI.aggregated_by("year", iris.analysis.MEAN))

                # find linear trend for ensemble members, JJA
                yx = tmax_UHI[m].data
                xx = tmax_UHI[m].coord("year").points
                xx = sm.add_constant(xx)
                estx = sm.OLS(yx, xx)
                estx = estx.fit()
                tmax_tall[n,m] = estx.params[1]       # trend

                yn = tmin_UHI[m].data
                xn = tmin_UHI[m].coord("year").points
                xn = sm.add_constant(xn)
                estn = sm.OLS(yn, xn)
                estn = estn.fit()
                tmin_tall[n,m] = estn.params[1]       # trend 
                '''
                # find linear trend for ensemble members, hottest
                if bias_corr and ngrids==2:
                    yx_hw = tmax_hw_UHI[m]+(bc_tmax_arr[m,tinds[0][0],tinds[0][1]]+bc_tmax_arr[m,tinds[1][0],tinds[1][1]])/2.-(bc_tmax_arr[m,binds[0][0],binds[0][1]]+bc_tmax_arr[m,binds[1][0],binds[1][1]])/2.
                else:
                    yx_hw = tmax_hw_UHI[m]
                x_hw = np.arange(1981,2080)
                x_hw = sm.add_constant(x_hw)
                estx_hw = sm.OLS(yx_hw, x_hw)
                estx_hw = estx_hw.fit()
                tmax_hw_tall[n,m] = estx_hw.params[1]  # trend

                if bias_corr and ngrids==2:
                    yn_hw = tmin_hw_UHI[m]+(bc_tmin_arr[m,tinds[0][0],tinds[0][1]]+bc_tmin_arr[m,tinds[1][0],tinds[1][1]])/2.-(bc_tmin_arr[m,binds[0][0],binds[0][1]]+bc_tmin_arr[m,binds[1][0],binds[1][1]])/2.
                else:
                    yn_hw = tmin_hw_UHI[m]
                estn_hw = sm.OLS(yn_hw, x_hw)
                estn_hw = estn_hw.fit()
                tmin_hw_tall[n,m] = estn_hw.params[1]  # trend                   
                '''
            # find linear trend for ensemble mean, JJA
            print("Finding ensemble-mean linear trends...")
            tmax_UHI_mean = sum(tmax_UHI)/len(tmax_UHI)
            yx = tmax_UHI_mean.data
            xx = tmax_UHI_mean.coord("year").points
            xx = sm.add_constant(xx)
            estx = sm.OLS(yx, xx)
            estx = estx.fit()
            tmax_trd[n] = estx.params[1]         # trend
            tmax_trd_pvals[n] = estx.pvalues[1]  # p value

            tmin_UHI_mean = sum(tmin_UHI)/len(tmin_UHI)
            yn = tmin_UHI_mean.data
            xn = tmin_UHI_mean.coord("year").points
            xn = sm.add_constant(xn)
            estn = sm.OLS(yn, xn)
            estn = estn.fit()
            tmin_trd[n] = estn.params[1]         # trend
            tmin_trd_pvals[n] = estn.pvalues[1]  # p value           
            '''
            # find linear trend for ensemble mean, hottest
            yx_hw = np.mean(tmax_hw_UHI, axis=0)
            estx_hw = sm.OLS(yx_hw, x_hw)
            estx_hw = estx_hw.fit()
            tmax_hw_trd[n] = estx_hw.params[1]          # trend   
            tmax_hw_trd_pvals[n] = estx_hw.pvalues[1]   # p value

            yn_hw = np.mean(tmin_hw_UHI, axis=0)
            estn_hw = sm.OLS(yn_hw, x_hw)
            estn_hw = estn_hw.fit()
            tmin_hw_trd[n] = estn_hw.params[1]          # trend      
            tmin_hw_trd_pvals[n] = estn_hw.pvalues[1]   # p value        
            '''
            # find trend ensemble range 
            tmax_tlh[n,0] = tmax_tall[n,:].min()                  # lower limit
            tmax_tlh[n,1] = tmax_tall[n,:].max()                  # upper limit
            tmin_tlh[n,0] = tmin_tall[n,:].min()                  # lower limit
            tmin_tlh[n,1] = tmin_tall[n,:].max()                  # upper limit  
            '''
            tmax_hw_tlh[n,0] = tmax_hw_tall[n,:].min()            # lower limit
            tmax_hw_tlh[n,1] = tmax_hw_tall[n,:].max()            # upper limit
            tmin_hw_tlh[n,0] = tmin_hw_tall[n,:].min()            # lower limit
            tmin_hw_tlh[n,1] = tmin_hw_tall[n,:].max()            # upper limit
            '''
            
            # plot time series
            yrs = tmax_UHI[0].coord("year").points
            #yrs = x_hw[:,1]
            print("Plotting tasmax & tasmin...")
            for m in range(12):
                axarr[spy[p], spx[p]].plot(yrs, tmax_UHI[m].data, color="pink", alpha=1.0)          # JJA: tmax_UHI[m].data; HW: tmax_hw_UHI[m]
                axarr[spy[p], spx[p]].plot(yrs, tmin_UHI[m].data, color="lightskyblue", alpha=1.0)  # JJA: tmin_UHI[m].data; HW: tmin_hw_UHI[m]
            axarr[spy[p], spx[p]].plot(yrs, yx, linewidth=3, color="red", label="day")              # JJA: yx; HW: yx_hw
            axarr[spy[p], spx[p]].plot(yrs, yn, linewidth=3, color="blue", label="night")           # JJA: yn; HW: yn_hw
            axarr[spy[p], spx[p]].axhline(y=0, color="k", linestyle=":", linewidth=2)
            axarr[spy[p], spx[p]].set_title(regs[n])
            axarr[spy[p], spx[p]].tick_params('both', length=12, width=2, which='major')
            axarr[spy[p], spx[p]].tick_params('both', length=8, width=1, which='minor')
            if bias_corr:
                axarr[spy[p], spx[p]].set_ylim([-2,4])   
                axarr[spy[p], spx[p]].yaxis.set_major_locator(FixedLocator([-2, 0, 2, 4]))
            else:
                axarr[spy[p], spx[p]].set_ylim([-1,4])                                     # JJA: [-1,4]; HW: [-1,6]
                axarr[spy[p], spx[p]].yaxis.set_major_locator(FixedLocator([0, 2, 4]))     # JJA: [0,2,4]; HW: [0, 2, 4, 6]
            axarr[spy[p], spx[p]].yaxis.set_minor_locator(MultipleLocator(0.5))
            axarr[spy[p], spx[p]].set_xlabel("Year")
            axarr[spy[p], spx[p]].set_xlim([1980, 2080])
            axarr[spy[p], spx[p]].xaxis.set_major_locator(FixedLocator([1980, 2000, 2020, 2040, 2060, 2080]))
            axarr[spy[p], spx[p]].xaxis.set_minor_locator(MultipleLocator(1))
            
            # just print everything
            print("1981-2079 time series of UHI...")
            print("tasmax JJA = ")
            print(yx)
            print("****")
            if bias_corr:
                np.save(out_dir+"ukcp18raw_"+regs[n]+"_"+str(ngrids)+"grids_biascorr_enmean_JJA_tasmax_UHI_evolution_"+str(syr)+"-"+str(eyr), yx)
            else:
                np.save(out_dir+"ukcp18raw_"+regs[n]+"_"+str(ngrids)+"grids_enmean_JJA_tasmax_UHI_evolution_"+str(syr)+"-"+str(eyr), yx)

            print("tasmin JJA  = ")
            print(yn)
            print("****")
            if bias_corr:
                np.save(out_dir+"ukcp18raw_"+regs[n]+"_"+str(ngrids)+"grids_biascorr_enmean_JJA_tasmin_UHI_evolution_"+str(syr)+"-"+str(eyr), yn)
            else:
                np.save(out_dir+"ukcp18raw_"+regs[n]+"_"+str(ngrids)+"grids_enmean_JJA_tasmin_UHI_evolution_"+str(syr)+"-"+str(eyr), yn)
            '''
            print("tasmax hw = ")
            print(yx_hw)
            print("****")

            print("tasmin hw = ")
            print(yn_hw)
            print("****")
            '''
        p += 1
    
    for ax in axarr.flat:
        ax.set(xlabel="Year")
    for ax in axarr.flat:
        ax.label_outer()

    f.text(0.01, 0.5, "UHI intensity ($^{\circ}$C)", va='center', rotation='vertical')

    # single legend
    handles, labels = axarr[0,0].get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    f.legend(handles, labels, loc="center right", borderaxespad=0.15)
    f.tight_layout()
    f.subplots_adjust(hspace=0.6, wspace=0.22, left=0.08, right=0.85)
        
    plt.savefig("/home/bridge/yl17544/plots/UKCP18_UHI_evolution/ukcp18raw_12km_top10_5by5_"+str(ngrids)+"grids_1981-2079_JJA_UHI_tasmax_tasmin_biascorr_dpi800.png", format="png", dpi=800)    # JJA: hottest3days; HW: hottest3days
    print("Saved UHI evolution figure!")    
    plt.close(f)
    
    # turn per year into per decade
    tmax_trd *= 10
    tmin_trd *= 10
    tmax_tall *= 10
    tmin_tall *= 10
    tmax_tlh *= 10
    tmin_tlh *= 10
    '''
    tmax_hw_trd *= 10
    tmin_hw_trd *= 10
    tmax_hw_tall *= 10
    tmin_hw_tall *= 10
    tmax_hw_tlh *= 10
    tmin_hw_tlh *= 10
    '''    
    # plot all trends
    x = np.arange(len(regs))
    f2, (ax2, ax3) = plt.subplots(2, 1, figsize=(18,10), num=2)
    for r in range(len(regs)):
        if (excl_regs[r]==False):
            # JJA
            '''
            # hatch according to ensemble mean p-value
            if tmax_trd_pvals[r] < 0.05:
                tmax_jja_pattern = None
            else:
                tmax_jja_pattern = "xxx"
            if tmin_trd_pvals[r] < 0.05:
                tmin_jja_pattern = None
            else:
                tmin_jja_pattern = "xxx" 
            # ensemble range error bars
            ax2.bar(x[r], tmax_trd[r], yerr=[[tmax_trd[r]-tmax_tlh[r,0]], [tmax_tlh[r,1]-tmax_trd[r]]], ecolor="black", width=0.3, color="red", hatch=tmax_jja_pattern, label="day")
            ax2.bar(x[r]+0.3, tmin_trd[r], yerr=[[tmin_trd[r]-tmin_tlh[r,0]], [tmin_tlh[r,1]-tmin_trd[r]]], ecolor="black", width=0.3, color="blue", hatch=tmin_jja_pattern, label="night")
            '''
            # hatch according to most ensemble members having the same sign
            tol = 1     # tolerance
            if np.sum((tmax_tall[r,:]/tmax_trd[r])<0) >= tol:
                tmax_jja_pattern = "//"
            else:
                tmax_jja_pattern = None
            if np.sum((tmin_tall[r,:]/tmin_trd[r])<0) >= tol:
                tmin_jja_pattern = "//"
            else:
                tmin_jja_pattern = None 
            # individual ensemble member trends
            ax2.bar(x[r], tmax_trd[r], ecolor="black", width=0.3, color="red", hatch=tmax_jja_pattern, label="day", zorder=1)          # bars
            ax2.scatter(np.repeat(x[r],12), tmax_tall[r,:], s=50, color="black", marker="x", zorder=2)                                 # crosses
            ax2.bar(x[r]+0.3, tmin_trd[r], ecolor="black", width=0.3, color="blue", hatch=tmin_jja_pattern, label="night", zorder=1)   # bar
            ax2.scatter(np.repeat(x[r]+0.3,12), tmin_tall[r,:], s=50, color="black", marker="x", zorder=2)                             # crosses          

            # hottest
            '''
            # hatch according to ensemble mean p-value
            if tmax_hw_trd_pvals[r] < 0.05:
                tmax_hw_pattern = None
            else:
                tmax_hw_pattern = "xxx"
            if tmin_hw_trd_pvals[r] < 0.05:
                tmin_hw_pattern = None
            else:
                tmin_hw_pattern = "xxx"
            # ensemble error bars
            ax3.bar(x[r], tmax_hw_trd[r], yerr=[[tmax_hw_trd[r]-tmax_hw_tlh[r,0]], [tmax_hw_tlh[r,1]-tmax_hw_trd[r]]], ecolor="black", width=0.3, color="red", hatch=tmax_hw_pattern, label="day")
            ax3.bar(x[r]+0.3, tmin_hw_trd[r], yerr=[[tmin_hw_trd[r]-tmin_hw_tlh[r,0]], [tmin_hw_tlh[r,1]-tmin_hw_trd[r]]], ecolor="black", width=0.3, color="blue", hatch=tmin_hw_pattern, label="night")
            '''
            '''
            if np.sum((tmax_hw_tall[r,:]/tmax_hw_trd[r])<0) >= tol:
                tmax_hw_pattern = "//"
            else:
                tmax_hw_pattern = None
            if np.sum((tmin_hw_tall[r,:]/tmin_hw_trd[r])<0) >= tol:
                tmin_hw_pattern = "//"
            else:
                tmin_hw_pattern = None
            # individual ensemble member trends
            ax3.bar(x[r], tmax_hw_trd[r], ecolor="black", width=0.3, color="red", hatch=tmax_hw_pattern, label="day", zorder=1)          # bars
            ax3.scatter(np.repeat(x[r],12), tmax_hw_tall[r,:], s=50, color="black", marker="x", zorder=2)                                 # crosses
            ax3.bar(x[r]+0.3, tmin_hw_trd[r], ecolor="black", width=0.3, color="blue", hatch=tmin_hw_pattern, label="night", zorder=1)   # bar
            ax3.scatter(np.repeat(x[r]+0.3,12), tmin_hw_tall[r,:], s=50, color="black", marker="x", zorder=2)                             # crosses   
            '''
        else:
            pass
    ax2.set_facecolor('white')
    ax2.spines['left'].set_position(('data',x[0]-0.3))
    ax2.spines['right'].set_position(('data',x[-1]+0.6))
    ax2.spines['bottom'].set_position('zero')
    ax2.spines['left'].set_color('black')
    ax2.spines['right'].set_color('black')
    ax2.spines['bottom'].set_color('black')
    ax2.spines['top'].set_color('none')
    ax2.set_xticks(x+0.2, minor=False)
    ax2.tick_params('x', length=0, width=0, which='major', pad=100)
    ax2.set_xticklabels([])
    ax2.set_xlim(x[0]-0.3, x[-1]+0.6)
    ax2.set_ylim(-0.1, 0.1)
    ax2.yaxis.set_major_locator(FixedLocator([-0.1, -0.05, 0, 0.05, 0.1]))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax2.tick_params('y', length=15, width=2, which='major')
    ax2.tick_params('y', length=10, width=1, which='minor')
    ax2.set_title("JJA")

    # customise legend with patches
    legend_elements = [Patch(facecolor="red", edgecolor="red", label="day"), Patch(facecolor="blue", edgecolor="blue", label="night")]
    ax2.legend(handles=legend_elements, prop={'size': 24}, loc="lower right", bbox_to_anchor=(0.98, -0.2), fancybox=True, framealpha=0.8, facecolor="white") 
    '''
    ax3.set_facecolor('white')
    ax3.spines['left'].set_position(('data',x[0]-0.3))
    ax3.spines['right'].set_position(('data',x[-1]+0.6))
    ax3.spines['bottom'].set_position('zero')
    ax3.spines['left'].set_color('black')
    ax3.spines['right'].set_color('black')
    ax3.spines['bottom'].set_color('black')
    ax3.spines['top'].set_color('none')
    ax3.set_xticks(x+0.2, minor=False)
    ax3.tick_params('x', length=0, width=0, which='major', pad=100)
    ax3.set_xticklabels(regs, rotation=90, fontdict={'horizontalalignment':"center"}, minor=False)
    ax3.set_xlim(x[0]-0.3, x[-1]+0.6)
    ax3.set_ylim(-0.1, 0.1)
    ax3.yaxis.set_major_locator(FixedLocator([-0.1, -0.05, 0, 0.05, 0.1]))
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))     
    ax3.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax3.tick_params('y', length=15, width=2, which='major')
    ax3.tick_params('y', length=10, width=1, which='minor')
    ax3.set_title("Warmest days")
    '''
    f2.text(0.01, 0.58, "UHI intensity trend ($^{\circ}$C per decade)", va='center', rotation='vertical')

    plt.tight_layout(rect=(0.02, 0.02, 1, 1))
    if bias_corr:
        plt.savefig("/home/bridge/yl17544/plots/UKCP18_UHI_evolution/ukcp18raw_12km_top10_5by5_"+str(ngrids)+"grids_1981-2079_JJA_hottest3days_UHI_tasmax_tasmin_trends_biascorr_all_hatch1outlier_dpi800.png", format="png", dpi=800)
    else:
        plt.savefig("/home/bridge/yl17544/plots/UKCP18_UHI_evolution/ukcp18raw_12km_top10_5by5_"+str(ngrids)+"grids_1981-2079_JJA_hottest3days_UHI_tasmax_tasmin_trends_all_hatch1outlier_dpi800.png", format="png", dpi=800)
    print("Saved UHI trends figure!")
    plt.close(f2)
    #plt.show()
     
    # print results
    print("UHI trends, London to Bristol...")
    #print("tasmax JJA ensemble-mean trends & all members & ensemble limits = ")
    #print(tmax_trd)
    #print(tmax_tall)
    #print(tmax_tlh)
    print("****")
    if bias_corr:
        np.save(out_dir+"ukcp18raw_top10_"+str(ngrids)+"grids_biascorr_enmean_JJA_tasmax_UHI_trends_"+str(syr)+"-"+str(eyr), tmax_trd)
        np.save(out_dir+"ukcp18raw_top10_"+str(ngrids)+"grids_biascorr_allmems_JJA_tasmax_UHI_trends_"+str(syr)+"-"+str(eyr), tmax_tall)
        np.save(out_dir+"ukcp18raw_top10_"+str(ngrids)+"grids_biascorr_lowhighlimits_JJA_tasmax_UHI_trends_"+str(syr)+"-"+str(eyr), tmax_tlh)
    else:
        np.save(out_dir+"ukcp18raw_top10_"+str(ngrids)+"grids_enmean_JJA_tasmax_UHI_trends_"+str(syr)+"-"+str(eyr), tmax_trd)
        np.save(out_dir+"ukcp18raw_top10_"+str(ngrids)+"grids_allmems_JJA_tasmax_UHI_trends_"+str(syr)+"-"+str(eyr), tmax_tall)
        np.save(out_dir+"ukcp18raw_top10_"+str(ngrids)+"grids_lowhighlimits_JJA_tasmax_UHI_trends_"+str(syr)+"-"+str(eyr), tmax_tlh)

    #print("tasmin JJA ensemble-mean trends & all members & ensemble limits = ")
    #print(tmin_trd)
    #print(tmin_tall)
    #print(tmin_tlh)
    print("****")
    if bias_corr:
        np.save(out_dir+"ukcp18raw_top10_"+str(ngrids)+"grids_biascorr_enmean_JJA_tasmin_UHI_trends_"+str(syr)+"-"+str(eyr), tmin_trd)
        np.save(out_dir+"ukcp18raw_top10_"+str(ngrids)+"grids_biascorr_allmems_JJA_tasmin_UHI_trends_"+str(syr)+"-"+str(eyr), tmin_tall)
        np.save(out_dir+"ukcp18raw_top10_"+str(ngrids)+"grids_biascorr_lowhighlimits_JJA_tasmin_UHI_trends_"+str(syr)+"-"+str(eyr), tmin_tlh)
    else:
        np.save(out_dir+"ukcp18raw_top10_"+str(ngrids)+"grids_enmean_JJA_tasmin_UHI_trends_"+str(syr)+"-"+str(eyr), tmin_trd)
        np.save(out_dir+"ukcp18raw_top10_"+str(ngrids)+"grids_allmems_JJA_tasmin_UHI_trends_"+str(syr)+"-"+str(eyr), tmin_tall)
        np.save(out_dir+"ukcp18raw_top10_"+str(ngrids)+"grids_lowhighlimits_JJA_tasmin_UHI_trends_"+str(syr)+"-"+str(eyr), tmin_tlh)
    '''
    #print("tasmax hw ensemble-mean trends & all members & ensemble limits = ")
    #print(tmax_hw_trd)
    print(tmax_hw_tall)
    #print(tmax_hw_tlh)
    print("****")
    if bias_corr:
        #np.save(out_dir+"ukcp18raw_top10_"+str(ngrids)+"grids_biascorr_enmean_hottest3days_tasmax_UHI_trends_"+str(syr)+"-"+str(eyr), tmax_hw_trd)
        np.save(out_dir+"ukcp18raw_top10_"+str(ngrids)+"grids_biascorr_allmems_hottest3days_tasmax_UHI_trends_"+str(syr)+"-"+str(eyr), tmax_hw_tall)
        #np.save(out_dir+"ukcp18raw_top10_"+str(ngrids)+"grids_biascorr_lowhighlimits_hottest3days_tasmax_UHI_trends_"+str(syr)+"-"+str(eyr), tmax_hw_tlh)
    else:
        #np.save(out_dir+"ukcp18raw_top10_"+str(ngrids)+"grids_enmean_hottest3days_tasmax_UHI_trends_"+str(syr)+"-"+str(eyr), tmax_hw_trd)
        np.save(out_dir+"ukcp18raw_top10_"+str(ngrids)+"grids_allmems_hottest3days_tasmax_UHI_trends_"+str(syr)+"-"+str(eyr), tmax_hw_tall)
        #np.save(out_dir+"ukcp18raw_top10_"+str(ngrids)+"grids_lowhighlimits_hottest3days_tasmax_UHI_trends_"+str(syr)+"-"+str(eyr), tmax_hw_tlh)

    #print("tasmin hw emsemble-mean trends & all members & ensemble limits = ")
    #print(tmin_hw_trd)
    print(tmin_hw_tall)
    #print(tmin_hw_tlh)
    print("****")
    if bias_corr:
        #np.save(out_dir+"ukcp18raw_top10_"+str(ngrids)+"grids_biascorr_enmean_hottest3days_tasmin_UHI_trends_"+str(syr)+"-"+str(eyr), tmin_hw_trd)
        np.save(out_dir+"ukcp18raw_top10_"+str(ngrids)+"grids_biascorr_allmems_hottest3days_tasmin_UHI_trends_"+str(syr)+"-"+str(eyr), tmin_hw_tall)
        #np.save(out_dir+"ukcp18raw_top10_"+str(ngrids)+"grids_biascorr_lowhighlimits_hottest3days_tasmin_UHI_trends_"+str(syr)+"-"+str(eyr), tmin_hw_tlh)
    else:
        #np.save(out_dir+"ukcp18raw_top10_"+str(ngrids)+"grids_enmean_hottest3days_tasmin_UHI_trends_"+str(syr)+"-"+str(eyr), tmin_hw_trd)
        np.save(out_dir+"ukcp18raw_top10_"+str(ngrids)+"grids_allmems_hottest3days_tasmin_UHI_trends_"+str(syr)+"-"+str(eyr), tmin_hw_tall)
        #np.save(out_dir+"ukcp18raw_top10_"+str(ngrids)+"grids_lowhighlimits_hottest3days_tasmin_UHI_trends_"+str(syr)+"-"+str(eyr), tmin_hw_tlh)
    
    print("UHI ensemble mean trend p values, London to Bristol...")
    print("tasmax JJA trend p values = ")
    print(tmax_trd_pvals)
    print("****")
    if bias_corr:
        np.save(out_dir+"ukcp18raw_top10_"+str(ngrids)+"grids_biascorr_enmean_JJA_tasmax_UHI_trend_pvals_"+str(syr)+"-"+str(eyr), tmax_trd_pvals)
    else:
        np.save(out_dir+"ukcp18raw_top10_"+str(ngrids)+"grids_enmean_JJA_tasmax_UHI_trend_pvals_"+str(syr)+"-"+str(eyr), tmax_trd_pvals)

    print("tasmin JJA trend p values = ")
    print(tmin_trd_pvals)
    print("****")
    if bias_corr:
        np.save(out_dir+"ukcp18raw_top10_"+str(ngrids)+"grids_biascorr_enmean_JJA_tasmin_UHI_trend_pvals_"+str(syr)+"-"+str(eyr), tmin_trd_pvals)
    else:
        np.save(out_dir+"ukcp18raw_top10_"+str(ngrids)+"grids_enmean_JJA_tasmin_UHI_trend_pvals_"+str(syr)+"-"+str(eyr), tmin_trd_pvals)

    print("tasmax hw trend p values = ")
    print(tmax_hw_trd_pvals)
    print("****")
    if bias_corr:
        np.save(out_dir+"ukcp18raw_top10_"+str(ngrids)+"grids_biascorr_enmean_hottest3days_tasmax_UHI_trend_pvals_"+str(syr)+"-"+str(eyr), tmax_hw_trd_pvals)
    else:
        np.save(out_dir+"ukcp18raw_top10_"+str(ngrids)+"grids_enmean_hottest3days_tasmax_UHI_trend_pvals_"+str(syr)+"-"+str(eyr), tmax_hw_trd_pvals)

    print("tasmin hw trend p values = ")
    print(tmin_hw_trd_pvals)
    print("****")
    if bias_corr:
        np.save(out_dir+"ukcp18raw_top10_"+str(ngrids)+"grids_biascorr_enmean_hottest3days_tasmin_UHI_trend_pvals_"+str(syr)+"-"+str(eyr), tmin_hw_trd_pvals)
    else:
        np.save(out_dir+"ukcp18raw_top10_"+str(ngrids)+"grids_enmean_hottest3days_tasmin_UHI_trend_pvals_"+str(syr)+"-"+str(eyr), tmin_hw_trd_pvals)
    '''
