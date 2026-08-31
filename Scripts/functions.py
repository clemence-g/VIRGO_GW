#IMPORT

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, date
from tqdm.notebook import tqdm
from scipy.signal import periodogram
import time

#SISMO
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy import read
from obspy import Stream
import os
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import calendar
import holidays

#GWOSC
import requests
from gwosc.datasets import event_detectors
from gwpy.timeseries import TimeSeries
from gwpy.time import from_gps
from astropy.time import Time
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns

#Réglage matplotlib pour ne pas avoir les symboles en double dans la légende 
import matplotlib as mpl
mpl.rcParams["legend.numpoints"] = 1
mpl.rcParams["legend.scatterpoints"] = 1

#METEO
import xarray as xr

#########################
######### SISMO #########
#########################

def dates_between(start_date, end_date,h = 24):
    dates = []
    current = start_date
    
    while current <= end_date:
        dates.append(current)
        current += timedelta(hours=h)
    
    return dates
client = Client("INGV") 

def download_day(dd,mm,yyyy, network = "VR", station = "VRG01", channel = "HH1"):
    filename = f"./{station}_{yyyy}_{mm}_{dd}.mseed"

    if os.path.exists(filename):
        print(f"Données déjà téléchargées : {filename}")
    else:
        try:
            st = client.get_waveforms(
                network=network,
                station=station,
                location="--",
                channel=channel,
                starttime=UTCDateTime(yyyy, mm, dd, 00, 00, 00, 000000),
                endtime=UTCDateTime(yyyy, mm, dd, 23, 59, 59, 999999))
            st.write(filename)

            print(filename)

        except Exception as e:
            print("Pas de données disponibles :", e)

def download_period(y, m1, d1, m2, d2, network = "VR", station = "VRG01", channel = "HH1"):
    """
    Télécharge les données sismologiques d'une station sur une période donnée.

    Les données sont récupérées jour par jour via le client FDSN d'INGV et
    enregistrées localement au format MiniSEED.

    Paramètres
    ----------
    y : int
        Année de la période à télécharger.
    m1, d1 : int
        Mois et jour de début.
    m2, d2 : int
        Mois et jour de fin.
    network : str, optionnel
        Code du réseau sismologique. Par défaut, ``"VR"``.
    station : str, optionnel
        Code de la station. Par défaut, ``"VRG01"``.
    channel : str, optionnel
        Code du canal sismologique. Par défaut, ``"HH1"``.

    Notes
    -----
    Les fichiers sont enregistrés dans le répertoire courant sous la forme
    ``station_année_mois_jour.mseed``.
    """
    days = dates_between(date(y,m1,d1),date(y,m2,d2))

    for d in tqdm(days):
        print(d)
        download_day(d.day,d.month,d.year, network, station, channel)

def download_year(y, network = "VR", station = "VRG01", channel = "HH1"):
    download_period(y, 1, 1, 12, 31, network, station, channel)

def remove_response(inventory, sta,cha,d,m,y):    
    filename = f"./{sta}_{y}_{m}_{d}.mseed"
    outfile = f"./{sta}_{y}_{m}_{d}_RR.mseed"

    if os.path.exists(outfile):
        print("Déjà corrigé :", outfile)
    else:
        try :
            st = read(filename)
            for tr in st:
                tr.data = tr.data.astype(np.float32)

            st.merge(fill_value="interpolate")

            st.remove_response(inventory,
                output="VEL",
                water_level=60,
                pre_filt=(0.01, 0.02, 40, 50))
            
            print("Réponse instrumentale retirée")
            st.write(outfile,format="MSEED")
            print(f"Saved as ", outfile)

        except FileNotFoundError:
            print("Pas de données")     
    
def remove_response_period(net, sta, cha, y, m1,d1,m2,d2):
    days = dates_between(date(y,m1,d1),date(y,m2,d2))
    for d in tqdm(days) :
        inventory = client.get_stations(
            network=net,
            station=sta,
            location="",
            channel=cha,
            level="response",
            starttime=UTCDateTime(d.year,d.month,d.day),
            endtime=UTCDateTime(d.year,d.month,d.day,23,59,59))
        
        print(d)
        remove_response(inventory, sta,cha,d.day,d.month,d.year)

def remove_response_year(net, sta, cha, y): 
    m1 = 1
    d1=1
    m2=12
    d2=31
    remove_response_period(net, sta, cha, y, m1,d1,m2,d2)

def cut_in_h(stream):
    tr = stream[0]
    stream_list = []
    current_time = tr.stats.starttime
    
    for i in range(24):
        end_time = current_time + 3600
        st = Stream([tr.slice(current_time, end_time)])
        stream_list.append(st)
        current_time=end_time
        
    return stream_list

def interpolate_nans(x):
    x = x.astype(float).copy()

    nans = np.isnan(x) 

    if nans.any():
        x[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), x[~nans]) 
    if np.all(nans):
        return x
    return x
    
def pad_to_full_hour(st_hour):
    tr = st_hour[0]
    x = tr.data
    fs = tr.stats.sampling_rate

    expected_len = int(3600 * fs)
    current_len = len(x)

    if current_len < expected_len:
        pad_len = expected_len - current_len
        x = np.concatenate([x, np.full(pad_len, np.nan)])

    return x, fs

def get_periodogram(st, title="PSD", plot=False):
    """
    Calcule la densité spectrale de puissance (PSD) d'une heure de données sismologiques.

    Les données sont complétées à une heure complète si nécessaire, puis les
    valeurs manquantes sont interpolées lorsque la complétude des données est
    suffisante. La PSD est calculée à l'aide d'un périodogramme et exprimée
    en décibels.

    Paramètres
    ----------
    st : obspy.Stream
        Données sismologiques correspondant à une fenêtre temporelle.
    title : str, optionnel
        Titre utilisé si la PSD est représentée graphiquement.
    plot : bool, optionnel
        Si ``True``, affiche la PSD calculée.

    Returns
    -------
    freqs : numpy.ndarray
        Fréquences associées à la PSD, limitées à 50 Hz.
    psd : numpy.ndarray
        Densité spectrale de puissance en dB.
    """
    x, fs = pad_to_full_hour(st)
    nan_fraction = np.isnan(x).mean()

    completeness = 1 - nan_fraction
    if 0.5 < completeness < 1:
        x = interpolate_nans(x)

    freqs, psd = periodogram(
        x,
        fs=fs,
        nfft=16384,
        window="boxcar",
        detrend="constant",
        return_onesided=True,
        scaling="density")

    psd = 10 * np.log10(psd)

    if plot:
        plt.semilogx(freqs, psd)
        plt.title(title)
        plt.xlabel("Fréquences (Hz)")
        plt.ylabel("PSD (dB)")

    mask = freqs <= 50
    psd = psd.astype(np.float32)
    return freqs[mask], psd[mask]

def PSD_by_h(stream): 
    psd_list = []
    date_list = []
    freq_list = None
    
    st_list = cut_in_h(stream)
    
    for i, st_hour in enumerate(st_list):
        if len(st_hour[0].data) == 0:
            continue
        date = st_hour[0].stats.starttime.datetime
        date_list.append(date)
 
        freq_list, psd = get_periodogram(
                st_hour,
                title=f"PSD from {i}:00 to {i+1}:00",
                plot=False
                )

        psd_list.append(psd)

    return np.array(psd_list), freq_list, date_list

def psd_period(y, m1, d1, m2, d2,station="VRG01", remove_response = False):
    """
    Calcule les PSD horaires des données sismologiques sur une période donnée.

    Les données MiniSEED de chaque journée sont chargées, divisées en fenêtres
    horaires, puis une PSD est calculée pour chaque fenêtre. Les résultats sont
    regroupés et sauvegardés dans un fichier NPZ.

    Paramètres
    ----------
    y : int
        Année de la période à analyser.
    m1, d1 : int
        Mois et jour de début.
    m2, d2 : int
        Mois et jour de fin.
    station : str, optionnel
        Code de la station sismologique.
    remove_response : bool, optionnel
        Indique si les fichiers dont la réponse instrumentale a été retirée
        doivent être utilisés.

    Returns
    -------
    all_psd : numpy.ndarray
        Tableau contenant une PSD par heure.
    all_dates : list
        Dates correspondant à chaque PSD.
    freqs : numpy.ndarray
        Fréquences associées aux PSD.
    """
    days = dates_between(date(y,m1,d1), date(y,m2,d2))
    all_psd = []
    all_dates = []

    for d in tqdm(days):
        print("\n",d)
        try:
            print("    Lecture des données...")
            if remove_response==True:
                filename = f"{station}_{d.year}_{d.month}_{d.day}_RR.mseed"
            else:
                filename = f"{station}_{d.year}_{d.month}_{d.day}.mseed" 
            st = read(filename)
            for tr in st:
                tr.data = tr.data.astype(float)
            st.merge(fill_value=np.nan)
            print(st)
            
            print("    Création des PSD...")
            psd_array, freqs, dates = PSD_by_h(st)

            all_psd.append(psd_array)
            all_dates.extend(dates)

        except FileNotFoundError:
            print("    Pas de données")
    
    all_psd = np.concatenate(all_psd, axis=0)
    print("Fini")
    
    if remove_response==True:
        outfile = f"PSD_{station}_{y}-{m1}-{d1}_{y}-{m2}-{d2}_RR.npz"
    else:   
        outfile = f"PSD_{station}_{y}-{m1}-{d1}_{y}-{m2}-{d2}.npz"
        
    np.savez(
        outfile,
        all_psd=all_psd,
        all_dates=all_dates,
        freqs=freqs)
    print("Enregistré en tant que ",outfile)
    
    return all_psd,all_dates,freqs

def load_psd(filename):
    data=np.load(filename, allow_pickle=True)

    return (data["all_psd"], pd.to_datetime(data["all_dates"]), data["freqs"])

def extract_period(all_psd,all_dates,freqs,limx):
    a,b=limx

    mask = (all_dates >= a) &(all_dates <= b)

    psd_month = all_psd[mask]
    dates_month = all_dates[mask]

    return psd_month, dates_month, freqs

def spec_plot(all_psd,all_dates,freqs,title=None, limx=None,outfile=None):
    """
    Représente l'évolution temporelle du contenu spectral du bruit sismique.

    Les PSD sont représentées sous forme de spectrogramme, avec la fréquence
    en ordonnée et le temps en abscisse. L'axe des fréquences est logarithmique
    et est limité à 50 Hz.

    Paramètres
    ----------
    all_psd : numpy.ndarray
        PSD calculées pour les différentes fenêtres temporelles.
    all_dates : array-like
        Dates correspondant aux PSD.
    freqs : numpy.ndarray
        Fréquences associées aux PSD.
    title : str, optionnel
        Titre du graphique.
    limx : tuple, optionnel
        Intervalle temporel à représenter.
    outfile : str, optionnel
        Chemin du fichier dans lequel enregistrer la figure.
    """
    
    if limx:
        all_psd, all_dates, freqs = extract_period(all_psd,all_dates,freqs,limx)
        
    dates = pd.to_datetime(all_dates)

    mask = freqs <= 50 

    plt.figure(figsize=(13,6))
    plt.pcolormesh(
        dates,
        freqs[mask],
        all_psd[:,mask].T,
        shading="auto",
        vmin=-190,
        vmax=-80,
        cmap="viridis")

    plt.ylim(10**(-2), 50)
    plt.yscale("log")
    
    plt.colorbar(label="PSD (dB)")
    
    plt.xlabel("Date",size=15)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Fréquences (Hz)",size=15)
    
    if title :
        plt.title(title,size=25)
    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.show()
    
def add_weekends(start_date, end_date):
    y = start_date.year 
    it_holidays = holidays.Italy(years=y)
    
    current = start_date
    while current <= end_date:
        if current.weekday() == 5:  
            plt.axvspan(
                current,
                current + timedelta(days=2),
                color="lightgray",
                alpha=0.6,
                zorder=0)
        
        current += timedelta(days=1)

    for d in it_holidays :
        plt.axvspan(
                d,
                d + timedelta(days=1),
                color="lightpink",
                alpha=0.6,
                zorder=0)

def median_plot(all_psd, all_dates, freqs, f1, f2, labell = "VRG01", limx=None, limy=None,plot=True, we = False, eq=False,outfile=None):
    """
    Représente l'évolution temporelle de la médiane de la PSD dans une bande
    de fréquences donnée.

    Pour chaque fenêtre temporelle, la médiane de la PSD est calculée entre
    les fréquences ``f1`` et ``f2``. La fonction permet également de représenter
    les week-ends et les événements sismiques sur le même graphique.

    Paramètres
    ----------
    all_psd : numpy.ndarray
        PSD calculées pour les différentes fenêtres temporelles.
    all_dates : array-like
        Dates correspondant aux PSD.
    freqs : numpy.ndarray
        Fréquences associées aux PSD.
    f1, f2 : float
        Bornes inférieure et supérieure de la bande de fréquences étudiée.
    labell : str, optionnel
        Étiquette utilisée dans la légende.
    limx, limy : tuple, optionnel
        Limites de l'axe des abscisses et des ordonnées.
    plot : bool, optionnel
        Si ``True``, affiche le graphique.
    we : bool, optionnel
        Si ``True``, indique les week-ends et jours fériés.
    eq : bool, optionnel
        Si ``True``, indique les événements sismiques.
    outfile : str, optionnel
        Chemin du fichier dans lequel enregistrer la figure.

    Returns
    -------
    all_dates : array-like
        Dates correspondant aux PSD.
    median_psd : list
        Médiane de la PSD pour chaque date.
    """
    plt.figure(figsize=(10,6))

    if limy:
        c,d=limy
    if limx:
        a,b=limx
    else:
        a=all_dates[0]
        b=all_dates[-1]
        
    mask_xlim = (all_dates >= a) & (all_dates <= b)
    x = all_dates[mask_xlim]
    
    if we :
        add_weekends(x[0], x[-1])
    if eq :
        add_eq_med(x[0], x[-1])
        
    median_psd = []

    print("Calcul des médianes...")

    band = (freqs > f1) & (freqs < f2)

    for psd in tqdm(all_psd):
        median = np.median(psd[band])
        median_psd.append(median)

    if plot == True:
        plt.plot(all_dates, median_psd, label = labell,color="black")
        plt.xlim(a,b)
        add_legend(we,eq)
        
        plt.ylabel("Médiane de PSD (dB)", size = 15)
        plt.xlabel("Date", size=15)
        plt.xticks(rotation=45, ha="right")
        if limy:
            plt.ylim(c,d)
        plt.grid(True)
        plt.title(f"Évolution des médianes de PSD ({f1}-{f2} Hz) avec le temps", size=20)

        if outfile:
            plt.savefig(outfile, dpi=300, bbox_inches="tight")

        plt.show()

    return all_dates, median_psd

def get_medians_4(all_psd, all_dates, freqs):
    y_list=[]
    band_list=[(0.01,0.1),(0.1,1),(1,5),(5,50)]
    
    for band in band_list:
        freq_band = (freqs >band[0]) & (freqs < band[1])
        median_psd = []
        
        for psd in all_psd:
            median = np.median(psd[freq_band])
            median_psd.append(median)
            
        y_list.append(median_psd)
        
    return y_list
    
def median_4_bis(all_psd, all_dates, freqs,limx=None, we=False, eq = False, p=80 ,n=5, outfile=None):
    """
    Représente l'évolution temporelle de la PSD dans quatre bandes de fréquences.

    Les bandes étudiées sont 0.01–0.1 Hz, 0.1–1 Hz, 1–5 Hz et 5–50 Hz.
    Les périodes présentant un niveau de bruit supérieur au percentile
    ``p`` pendant au moins ``n`` heures consécutives sont mises en évidence.

    Paramètres
    ----------
    all_psd : numpy.ndarray
        PSD calculées pour les différentes fenêtres temporelles.
    all_dates : array-like
        Dates correspondant aux PSD.
    freqs : numpy.ndarray
        Fréquences associées aux PSD.
    limx : tuple, optionnel
        Intervalle temporel à représenter.
    we : bool, optionnel
        Si ``True``, indique les week-ends et jours fériés.
    eq : bool, optionnel
        Si ``True``, indique les événements sismiques.
    p : float, optionnel
        Percentile utilisé pour définir les anomalies.
    n : int, optionnel
        Nombre minimal d'heures consécutives nécessaires pour définir une
        période d'anomalie.
    outfile : str, optionnel
        Chemin du fichier dans lequel enregistrer la figure.
    """
    if limx:
        a,b=limx
    else:
        a=all_dates[0]
        b=all_dates[-1]
        
    mask_xlim = (all_dates >= a) & (all_dates <= b)
    x = all_dates[mask_xlim]

    y_list = get_medians_4(all_psd, all_dates, freqs)
    y1 = np.array(y_list[0])[mask_xlim]
    y2 = np.array(y_list[1])[mask_xlim]
    y3 = np.array(y_list[2])[mask_xlim]
    y4 = np.array(y_list[3])[mask_xlim]

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(9, 8), sharex=True)

    axes[0].plot(x, y1, color='blue')
    axes[0].grid(True)
    axes[0].set_ylabel("0.01 - 0.1 Hz")
    axes[0].set_ylim(-160,-100)
    axes[0].set_xlim(a,b)
    
    anomaly_periods_0 = get_anomaly(x, y1, p, n)
    for start, end in anomaly_periods_0:
        mask = (x>= start) & (x <= end)  
        axes[0].plot(
            np.array(x)[mask],
            y1[mask],
            color="black",
            linewidth=2)

    axes[1].plot(x, y2, color='orange')
    axes[1].grid(True)
    axes[1].set_ylabel("0.1 - 1 Hz")
    axes[1].set_xlim(a,b)
    axes[1].set_ylim(-160,-130)
    
    anomaly_periods_1 = get_anomaly(x, y2, p, n)
    for start, end in anomaly_periods_1:
        mask = (x>= start) & (x <= end)
    
        axes[1].plot(
            np.array(x)[mask],
            y2[mask],
            color="black",
            linewidth=2)

    axes[2].plot(x, y3, color='green')
    axes[2].grid(True)
    axes[2].set_ylabel("1 - 5 Hz")
    axes[2].set_xlim(a,b)
    axes[2].set_ylim(-170,-145)
        
    anomaly_periods_2 = get_anomaly(x, y3, p, n)
    for start, end in anomaly_periods_2:
        mask = (x>= start) & (x <= end)
    
        axes[2].plot(
            np.array(x)[mask],
            y3[mask],
            color="black",
            linewidth=2)

    axes[3].plot(x, y4, color='red')
    axes[3].grid(True)
    axes[3].set_xlabel("Date")
    axes[3].tick_params(axis="x", rotation=45)
    axes[3].set_ylabel("5 - 50 Hz")
    axes[3].set_xlim(a,b)
    axes[3].set_ylim(-180,-155)
    
    anomaly_periods_3 = get_anomaly(x, y4, p, n)
    for start, end in anomaly_periods_3:
        mask = (x>= start) & (x <= end)
    
        axes[3].plot(
            np.array(x)[mask],
            y4[mask],
            color="black",
            linewidth=2)

    plt.tight_layout()

    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches="tight")

    plt.show()


def get_eq(start, end):
    """
    Récupère les événements sismiques autour du détecteur Virgo.

    Les séismes sont répartis en trois catégories selon leur distance au site
    de Virgo : locaux, régionaux et globaux. Des seuils de magnitude différents
    sont utilisés pour chaque catégorie.

    Paramètres
    ----------
    start, end : datetime-like
        Début et fin de la période recherchée.

    Returns
    -------
    mag_local, time_local : list
        Magnitudes et dates des séismes locaux.
    mag_regional, time_regional : list
        Magnitudes et dates des séismes régionaux.
    mag_global, time_global : list
        Magnitudes et dates des séismes globaux.
    """
    lat_virgo = 43.6314
    lon_virgo = 10.5045
    
    events_local= Client("INGV").get_events(
        latitude=lat_virgo,
        longitude=lon_virgo,
        maxradius=2.7,
        starttime=start,
        endtime=end,
        minmagnitude = 2.5)

    time_local = [events_local[i].origins[0].time for i in range(len(events_local))]
    mag_local = [events_local[i].magnitudes[0].mag for i in range(len(events_local))]
    
    events_regional= Client("USGS").get_events(
        latitude=lat_virgo,
        longitude=lon_virgo,
        minradius = 2.7,
        maxradius=9,
        starttime=start,
        endtime=end,
        minmagnitude = 4.5)

    time_regional = [events_regional[i].origins[0].time for i in range(len(events_regional))]
    mag_regional = [events_regional[i].magnitudes[0].mag for i in range(len(events_regional))]

    events_global= Client("USGS").get_events(
        latitude=lat_virgo,
        longitude=lon_virgo,
        minradius = 9,
        maxradius=180,
        starttime=start,
        endtime=end,
        minmagnitude = 6.5)

    time_global = [events_global[i].origins[0].time for i in range(len(events_global))]
    mag_global = [events_global[i].magnitudes[0].mag for i in range(len(events_global))]

    return mag_local,time_local, mag_regional,time_regional, mag_global,time_global

def add_eq_med(start_date, end_date):
    mag_local,time_local, mag_regional,time_regional, mag_global,time_global  = get_eq(start_date, end_date)

    ax = plt.gca()
    
    for mag,t in zip(mag_local,time_local):
            ax.axvline(
        [t.datetime],
        alpha=min(mag/8, 1),
        color="green",
        linestyle="dotted",label="_nolegend_" )
        
    for mag,t in zip(mag_regional,time_regional):
            ax.axvline(
        [t.datetime],
        alpha=min(mag/8, 1),
        color="green",
        linestyle="dashdot",label="_nolegend_")
        
    for mag,t in zip(mag_global,time_global):
            ax.axvline(
        [t.datetime ],
        alpha=min(mag/8, 1),
        color="green",
        linestyle="dashed",label="_nolegend_")
        
    return True

def add_legend(we, eq):
    if we or eq :
        handles, labels = plt.gca().get_legend_handles_labels()
        
    if we == False:
        if eq ==  False:
            plt.legend()
            
    if we:
        handles.append(
            mpatches.Patch(
                color="grey",
                alpha=0.3,
                label="Weekend"))
        
        handles.append(
            mpatches.Patch(
                color="lightpink",
                alpha=0.3,
                label="Jours fériés"))

    if eq:
        handles.append(
        mlines.Line2D(
            [0], [0],
            color="green",
            linestyle=":",
            linewidth=2,
            label="Séismes locaux \n(r<2.7° et mag >2.5)"
        )
    )

        handles.append(
        mlines.Line2D(
            [0], [0],
            color="green",
            linestyle="-.",
            linewidth=2,
            label="Séismes régionaux \n(2.7<r<9° et mag >4.5)"
        )
    )

        handles.append(
        mlines.Line2D(
            [0], [0],
            color="green",
            linestyle="--",
            linewidth=2,
            label="Séismes globaux \n(9<r<180° et mag >6.5)"
        )
    )
        handles.append(
            mlines.Line2D(
                [0, 1], [0, 0],
                linestyle="None",
                label=r"$\mathbf{Magnitude}$ :"
            )
        )

        handles.append(
            mlines.Line2D(
                [],
                [],
                marker="|",
                color="green",
                alpha=0.25,
                linestyle="None",
                markersize=12,
                markeredgewidth=2,
                label="Faible"
            )
        )

        handles.append(
            mlines.Line2D(
                [0, 1], [0, 0],
                marker="|",
                color="green",
                alpha=0.6,
                linestyle="None",
                markersize=12,
                markeredgewidth=2,
                label="Intermédiaire"
            )
        )

        handles.append(
            mlines.Line2D(
                [0, 1], [0, 0],
                marker="|",
                color="green",
                alpha=1,
                linestyle="None",
                markersize=12,
                markeredgewidth=2,
                label="Forte"
            )
        )
        
    if we or eq:
        plt.legend(handles=handles,loc="center left", bbox_to_anchor=(1, 0.5),fontsize=15)

def get_anomaly(dates,data, p, n):
    """
    Identifie les périodes présentant des valeurs anormalement élevées.

    Une anomalie est définie comme une séquence d'au moins ``n`` valeurs
    consécutives dépassant le percentile ``p`` de l'ensemble des données.

    Paramètres
    ----------
    dates : array-like
        Dates associées aux données.
    data : array-like
        Série temporelle à analyser.
    p : float
        Percentile utilisé comme seuil d'anomalie.
    n : int
        Nombre minimal de valeurs consécutives au-dessus du seuil.

    Returns
    -------
    anomaly_periods : list of tuple
        Liste des périodes d'anomalie sous la forme ``(date_debut, date_fin)``.
    """
    
    threshold = np.nanpercentile(data,p)
    data_above = data > threshold
    
    i = 0
    list_anomaly = []
    result = []
    
    while i < len(data_above):
        if data_above[i] == True :
            result.append(i)
            i+=1
        else :
            if len(result) >= n : 
                list_anomaly.append(result)
                
            result = []
            i+=1
            
    if len(result) >= n:
        list_anomaly.append(result)
    
    anomaly_periods = []

    if len(list_anomaly) >0:
        for anomaly in list_anomaly:
            start = dates[anomaly[0]]
            end = dates[anomaly[-1]]
            anomaly_periods.append((start, end))
            
    return anomaly_periods
            
    
#########################
######### GWOSC #########
#########################

def get_catalogs():
    url = "https://gwosc.org/eventapi/json/"
    r = requests.get(url)
    print(r.text)

def get_detectors(event):
    detectors_list = []
    real_name = list(event["events"].keys())[0]

    for strain in event['events'][real_name]['strain']:
        detectors_list.append(strain['detector'])
    
    detectors_name = set(detectors_list)
    
    return detectors_name

def day_or_night(date):
    date_utc = date.tz_localize("UTC")
    local_time = date_utc.tz_convert("Europe/Rome")
    if 7<=local_time.hour <19:
        result = "day"
    else:
        result = "night"
        
    return result

def get_detector_with_retry(url, max_attempts,i):
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            event = response.json()
            
            return get_detectors(event)

        except requests.exceptions.RequestException as e:
            print(f"Évenement {i} : Tentative {attempt+1}/{max_attempts} Ratée: {e}")

            if attempt < max_attempts - 1:
                time.sleep(2)  
    
    return set()
    
def get_data(catalog,instrument=None):
    """
    Récupère les événements d'un catalogue GWOSC et identifie les détecteurs
    ayant contribué à chaque événement.

    Les temps GPS sont convertis en dates UTC et, lorsqu'un instrument est
    spécifié, seuls les événements détectés par cet instrument sont conservés.

    Paramètres
    ----------
    catalog : str
        Identifiant du catalogue GWOSC.
    instrument : str, optionnel
        Code du détecteur à sélectionner, par exemple ``"V1"``.

    Returns
    -------
    pandas.DataFrame
        Tableau contenant les événements du catalogue, leurs propriétés,
        les détecteurs impliqués et, si un instrument est spécifié,
        leur classification jour/nuit.
    """
    
    print("Récupération des données...")
    url = f"https://gwosc.org/eventapi/json/{catalog}"
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data["events"]).T

    print("Conversion des GPS en dates...")
    df["date"] = Time(df["GPS"].astype(float),format="gps").datetime

    print("Spécification du détecteur...")
    detectors_list = []

    for i in tqdm(range(len(df))):
        detect = get_detector_with_retry(df.iloc[i]["jsonurl"],3,i)
        detectors_list.append(detect)

    df["detector"]=detectors_list

    print("Données prêtes")

    if instrument :
        print(f"Sorting by instrument {instrument}...")
        indices = []

        for i in range(len(df)):
            if instrument in df.iloc[i]["detector"]:
                indices.append(i)
                
        print("Spécification jour/nuit...")
        df_inst = df.iloc[indices].copy()
        df_inst["day_or_night_V1"] = df_inst["date"].apply(day_or_night)
        print("Fini !")
        
        return df_inst
    return df

def select_year(df,y):
    df = df[df["date"].dt.year == y]
    return df

def day_night_ratio(df):
    day = 0
    night = 0
    
    for _, row in df.iterrows():
        result = row['day_or_night_V1']
        if result=="day":
            day+=1
        elif result =="nuit":
            night+=1
            
    print(f"Il y a {day} évenements pendant la journée et {night} événements pendant la nuit.")
    
    if night!=0:
        print("Rapport jour/nuit :", day/night)

def plot_events(df,run,offline_periods=None, lim = None,outfile=None):
    """
    Représente les événements d'ondes gravitationnelles détectés au cours
    d'une campagne d'observation.

    La distance lumineuse est représentée en fonction du temps. La taille
    des marqueurs représente la masse totale de la source et leur opacité
    représente le rapport signal/bruit. Les périodes pendant lesquelles le
    détecteur n'est pas disponible peuvent également être indiquées.

    Paramètres
    ----------
    df : pandas.DataFrame
        Données des événements à représenter.
    run : str
        Nom de la campagne d'observation.
    offline_periods : list of tuple, optionnel
        Périodes pendant lesquelles le détecteur n'est pas disponible.
    lim : tuple, optionnel
        Limites temporelles du graphique.
    outfile : str, optionnel
        Chemin du fichier dans lequel enregistrer la figure.
    """
    
    plt.figure(figsize=(10, 6))
    #Retirer les ´´´ pour représenter les échelles de Laniakea et de l'univers observable
    '''
    if len(df)>0:
        plt.axhline(4231.0992362856, color= "black", ls = "--")
        plt.axhline(160, color= "black", ls = "--")
    
        plt.text(df.iloc[-1]["date"]+ pd.Timedelta(days=1), 4331,"Observable universe scale")

        plt.text(df.iloc[-1]["date"]+ pd.Timedelta(days=1), 260,"Laniakea scale")
    '''
    
    if offline_periods:
        for start, end in offline_periods:
            plt.axvspan(
                start,
                end,
                color="lightgray",
                alpha=0.5,
                zorder=0)
            
    default_mass = df["total_mass_source"].median(skipna=True)

    for _, row in df.iterrows():
        mass = row["total_mass_source"]
        if pd.isna(mass):
            mass = default_mass
        alpha = 0.2 + 0.8 * (
        (row["network_matched_filter_snr"] - df["network_matched_filter_snr"].min())
        / (df["network_matched_filter_snr"].max() - df["network_matched_filter_snr"].min()))

        plt.scatter(
            row["date"],
            row["luminosity_distance"],
            s=mass*5,
            color="orange" if row["day_or_night_V1"]=="day" else "dodgerblue",
            alpha=alpha)
    
    plt.grid(True)
    plt.xlabel("Date", size = 15)
    
    if lim :
        a,b = lim
        
    else:
        a = df["date"].iloc[-1]- timedelta(days=1)
        b = df["date"].iloc[0]+ timedelta(days=1)
        
    plt.xlim(a,b)

    legend_elements = [
    Line2D([], [], linestyle="None", label=r"$\mathbf{Jour/Nuit}$"),
    Line2D(
        [0], [0],
        marker='o',
        color='w',
        linestyle='',
        label='Jour',
        markerfacecolor='orange',
        markersize=10),
    Line2D(
        [0], [0],
        marker='o',
        color='w',
        linestyle='',
        label='Nuit',
        markerfacecolor='dodgerblue',
        markersize=10)
    ]

    legend_elements.extend([
    Line2D([], [], linestyle="None", label=r"$\mathbf{Masse}\text{ }\mathbf{totale}\text{ } \mathbf{de}\text{ } \mathbf{la}\text{ } \mathbf{source}$"),
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor='black',
           linestyle='',
           markersize=(5*10)**0.5,
           label='10 $M_\\odot$'),

    Line2D([0], [0], marker='o', color='w',
           markerfacecolor='black',
           markersize=(50*5)**0.5,
           label='50 $M_\\odot$'),

    Line2D([0], [0], marker='o', color='w',
           markerfacecolor='black',
           linestyle='',
           markersize=(100*5)**0.5,
           label='100 $M_\\odot$')
    ])

    if df.empty:
        min_snr = 7
        max_snr = 25
    else:
        min_snr =df["network_matched_filter_snr"].min()
        max_snr =df["network_matched_filter_snr"].max()
               
    legend_elements.extend([
    Line2D([], [], linestyle="None", label=r"$\mathbf{Rapport} \mathbf{ } \mathbf{signal/bruit}$"),
        
    Line2D([0], [0], marker='o', color='black', 
           alpha= 0.2 + 0.8 * ((min_snr - min_snr)/(max_snr - min_snr)),
           linestyle='',
           markersize=10, label=f"Min = {min_snr}"),

    Line2D([0], [0], marker='o', color='black',
           alpha=0.2 + 0.8 * ((np.round((max_snr+min_snr)/2) - min_snr)/(max_snr - min_snr)),
           linestyle='',
           markersize=10, label=f"Moyenne = {(max_snr+min_snr)/2}"),

    Line2D([0], [0], marker='o', color='black',
           alpha=0.2 + 0.8 * ((max_snr - min_snr)/(max_snr - min_snr)),
           linestyle='',
           markersize=10, label=f"Max = {max_snr}")
    ])
    
    legend_elements.extend([
    Line2D([], [], linestyle="None", label=r"$\mathbf{Statut}\ \mathbf{du}\ \mathbf{détecteur}$"),

    Patch(
        facecolor="lightgray",
        edgecolor="gray",
        alpha=0.5,
        label="Pas de données disponibles")
    ])
    
    plt.xticks(rotation=45, ha="right")
    plt.grid(True)
    plt.xlabel("Date", size = 15)
    plt.ylabel("Distance lumineuse (Mpc)", size = 15)

    plt.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5), labelspacing = 1, fontsize=10)

    plt.title(f'Série {run} du {a.date()} au {b.date()}', size=20)
    
    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches="tight")

    plt.show()

def events_by_month(df):

    month_number = np.zeros(12, dtype=int)

    for month in df["date"].dt.month:
        month_number[month - 1] += 1

    return month_number

def plot_months(df):
    month_number = events_by_month(df)
    months = [
    "Janvier", "Février", "Mars", "Avril",
    "Mai", "Juin", "Juillet", "Août",
    "Septembre", "Octobre", "Novembre", "Decembre"]
    
    plt.figure(figsize=(10, 6))

    plt.bar(months, month_number)
    plt.xlabel("Mois")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Nombre de détections")
    
    min_date = df["date"].iloc[-1]
    max_date = df["date"].iloc[0]
    plt.title(f"Nombre d'événements détectés par Virgo par mois \npour la période {min_date} - {max_date}", size = 15)
    plt.show()

def events_by_functionning_day(df, y, offline_periods):
    month_rate = []
    month_number = events_by_month(df)

    for m in range(1, 13):
        nb_days = calendar.monthrange(y, m)[1]
        missing_days = 0

        for period in offline_periods:
            start, end = period

            # période d'arrêt entièrement dans le mois
            if start.month == m and end.month == m:
                missing_days += len(dates_between(start, end))

            # arrêt qui commence dans le mois mais finit après
            elif start.month == m and end.month != m:
                missing_days += len(dates_between(start, datetime(y, m, nb_days)))

            # arrêt qui commence avant le mois mais finit dedans
            elif start.month != m and end.month == m:
                missing_days += len(dates_between(datetime(y, m, 1), end))

            # arrêt qui englobe tout le mois
            elif start < datetime(y, m, 1) and end > datetime(y, m, nb_days):
                missing_days += nb_days
        functioning_days = nb_days - missing_days

        if functioning_days > 0:
            rate = month_number[m-1] / functioning_days
        else:
            rate = 0

        month_rate.append(rate)

    return month_rate
    
def plot_rate(df,month_rate):
    months = [
    "Janvier", "Février", "Mars", "Avril",
    "Mai", "Juin", "Juillet", "Août",
    "Septembre", "Octobre", "Novembre", "Decembre"]
    
    plt.figure(figsize=(10, 6))

    plt.bar(months, month_rate, color = "dodgerblue")
    plt.xlabel("Mois")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Taux de détections par jour de fonctionnement")
    
    date_min = df["date"].iloc[-1]
    date_max = df["date"].iloc[0]
    plt.title(f"Taux de détections par jour de fonctionnement \npour la période {date_min}-{date_max}", size=15)
    plt.show()

def events_by_day(df):
    day_number = np.zeros(7, dtype=int)

    for d in df["date"].dt.weekday:
        day_number[d] += 1

    return day_number

def plot_days(df):
    day_number = events_by_day(df)
    days = [
    "Lundi", "Mardi", "Mercredi",
    "Jeudi", "Vendredi", "Samedi", "Dimanche"
    ] 
    
    plt.figure(figsize=(10, 6))
    plt.bar(days, day_number, color = "darkblue")
    plt.xlabel("Jour de la semaine")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Nombre de détections")
    
    min_date = df["date"].iloc[-1]
    max_date = df["date"].iloc[0]
    plt.title(f"Nombre de détections par Virgo par jour de la semaine \npour la période {min_date} - {max_date}", size=15)
    plt.show()

def process_data(run):
    """
    Récupère les données de sensibilité de Virgo disponibles sur GWOSC.

    Pour une campagne d'observation donnée, récupère les segments de données
    du détecteur Virgo et extrait notamment le duty cycle, les indicateurs
    BLRMS et le BNS range.

    Paramètres
    ----------
    run : str
        Campagne d'observation à traiter (par exemple ``"O3a"`` ou ``"O4b"``).

    Returns
    -------
    pandas.DataFrame
        Tableau contenant les intervalles temporels et les indicateurs de
        sensibilité associés à chaque segment.
    """
    
    run_url={
        "O2":"https://gwosc.org/archive/links/O2_4KHZ_R1/V1/1164556817/1187733618/json/",
        "O3a":"https://gwosc.org/archive/links/O3a_4KHZ_R1/V1/1238166018/1253977218/json/",
        "O3b":"https://gwosc.org/archive/links/O3b_4KHZ_R1/V1/1256655618/1269363618/json/",
        "O4b":"https://gwosc.org/archive/links/O4b_16KHZ_R1/V1/1396417050/1422118818/json/"}
    
    url =run_url[run]

    rows = []
    data = requests.get(url).json()

    for seg in data["strain"]:
        if seg["format"] != "hdf5":
            continue

        rows.append({
        "gps_start": seg["GPSstart"],
        "gps_end": seg["GPSstart"] + seg["duration"],
        "duration": seg["duration"],
        "Duty cycle": seg["duty_cycle"],
        "blrms200" : seg["BLRMS200"],
        "blrms1000" : seg["BLRMS1000"],
        "BNS range": seg["BNS"],
        })

    df = pd.DataFrame(rows)
    df["UTCstart"] = Time(df["gps_start"], format="gps").to_datetime()
    df["UTCend"] = Time(df["gps_end"], format="gps").to_datetime()
    
    return df

def plot_sensitivity(df,type="BNS range",xlim=None,ylim=None,we=False, eq = False,outfile=None):
    """
    Représente l'évolution temporelle d'un indicateur de sensibilité de Virgo.

    Les valeurs sont représentées sous forme de segments horizontaux
    correspondant aux intervalles de temps disponibles dans les données
    GWOSC. Les périodes sans données sont laissées vides.

    Paramètres
    ----------
    df : pandas.DataFrame
        Données de sensibilité produites par ``process_data``.
    type : str, optionnel
        Nom de la colonne à représenter, par exemple ``"BNS range"``.
    xlim, ylim : tuple, optionnel
        Limites temporelles et verticales du graphique.
    we : bool, optionnel
        Si ``True``, indique les week-ends et jours fériés.
    eq : bool, optionnel
        Si ``True``, indique les événements sismiques.
    outfile : str, optionnel
        Chemin du fichier dans lequel enregistrer la figure.
    """
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if xlim:
        a,b=xlim
    else:
        a = df["UTCstart"].min()
        b = df["UTCend"].max()

    df_new = df[(df["UTCend"] >= a) & (df["UTCstart"] <= b)]

    x = []
    y = []

    previous_end = None
    for _, row in df_new.iterrows():

            start = row["UTCstart"]
            end = row["UTCend"]
            value = row[type]

            if previous_end is not None and start > previous_end:
                x.append(previous_end)
                y.append(np.nan)

                x.append(start)
                y.append(np.nan)
            x.append(start)
            y.append(value)

            x.append(end)
            y.append(value)

            previous_end = end

    if we :
        add_weekends(a,b)
    if eq :
        eq=add_eq_med(df["UTCend"].min(),df["UTCend"].max())

    if type == "BNS range":
        label = "BNS range (Mpc)"
    if type == "Duty cycle":
        label = "Duty cycle (%)"
    ax.plot(x, y)
    ax.fill_between(x,y,alpha=0.3)
    ax.set_ylabel(label, size =15)
    ax.set_xlabel("Date", size=15)
    ax.set_title(f"Évolution temporelle du {label} \npour la période {a}-{b}", size=25)
    
    if ylim:
        c,d=ylim
        plt.ylim(c, d)
        
    if xlim:
        plt.xlim(a, b)
        
    add_legend(we,eq)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.show()

def get_median_profile(df,param_1,param_2,dt=0.1,window=1):
    param_range = np.arange(np.min(df[param_2]),np.max(df[param_2]),dt)
    median_profile = []

    for d in param_range:
        distance = np.abs(df[param_2] - d) 
        values = df.loc[distance < window,param_1]  
        median_profile.append(values.median()) 
        
    return param_range,median_profile

def plot_bns_param(df,param_1,param_2,mode = "median", degree = 3,xlim=None,ylim=None,dt=0.1,window=1, title=None,outfile=None):
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df,x=param_2,y=param_1, hue='day_or_night', palette={
        "day": "orange",
        "night": "dodgerblue"
    },legend="brief")
    
    if param_1 == "BNS range":
        label = "BNS range (Mpc)"
    elif param_1 == "Duty cycle":
        label = "Duty cycle (%)"

    if mode == "median":
        param_range,median_profile = get_median_profile(df,param_1,param_2,dt,window)

        plt.plot(
        param_range,
        median_profile,
        color="red",
        linewidth=2,
        label=f"Médiane glissante, dt = {dt}, window = {window}")
    elif mode == "reg":
        data = df[[param_2, param_1]].dropna()

        x = data[param_2].values
        y = data[param_1].values

        coefficients = np.polyfit(x, y, degree)

        x_reg = np.linspace(x.min(), x.max(), 500)

        y_reg = np.polyval(coefficients, x_reg)

        plt.plot(
        x_reg,
        y_reg,
        color="red",
        linewidth=2,
        label=f"Régression polynomiale (degré {degree})"
        )


    plt.xlabel(param_2, size = 15)
    plt.ylabel(label, size = 15)
    
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)
    plt.title(title, size=25)
    plt.grid()
    plt.legend()
    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.show()

#STRAIN TIMESERIES
def cut_in_600_timeseries(strain, start, end):
    strain_list = []
    current_time = start

    while current_time < end:

        next_time = current_time + 600

        st_hour = strain.crop(
            current_time,
            min(next_time, end))

        strain_list.append(st_hour)

        current_time = next_time

    return strain_list

def pad_to_600_timeseries(strain_hour):
    x = strain_hour.value
    fs = strain_hour.sample_rate.value

    expected_len = int(600 * fs)

    current_len = len(x)

    if current_len < expected_len:
        pad_len = expected_len - current_len
        x = np.concatenate([x, np.full(pad_len, np.nan)])

    return x, fs

def get_periodogram_strain(strain_hour, title="PSD", plot=False, fmax=1000):
    x, fs = pad_to_600_timeseries(strain_hour)

    nan_fraction = np.isnan(x).mean()
    completeness = 1 - nan_fraction
    if completeness > 0.5:
        x = interpolate_nans(x)

    freqs, psd = periodogram(
        x,
        fs=fs,
        window="hann",
        nfft=262144,
        detrend="constant",
        scaling="density"
        )
    
    psd = 10*np.log10(psd)
    
    mask = ((freqs > 0) & (freqs <= fmax))

    if plot:
        plt.figure(figsize=(8,5))
        plt.semilogx(freqs[mask],psd[mask])
        plt.xlabel("Fréquence (Hz)")
        plt.ylabel("PSD (dB)")
        plt.title(title)
        plt.grid()
        plt.show()

    return freqs[mask], psd[mask]

def PSD_strain_by_h(strain, start, end):
    psd_list = []
    date_list = []
    freq_list = None

    strain_hours = cut_in_600_timeseries(strain, start, end)

    for i, strain_hour in enumerate(tqdm(strain_hours)):
        freqs, psd = get_periodogram_strain(
            strain_hour,
            title=f"PSD heure {i}",
            plot=False)
        
        if psd is None:
            continue

        psd_list.append(psd)
        date_list.append(from_gps(strain_hour.t0.value))

        freq_list = freqs

    return (np.array(psd_list), np.array(freq_list), pd.to_datetime(date_list, utc=True))

def psd_strain_period(detector, gps_start, gps_end):
    """
    Télécharge les données de strain et calcule leurs PSD sur une période donnée.

    Les données sont récupérées depuis GWOSC par fenêtres de 10 minutes.
    Une PSD est calculée pour chaque fenêtre et les résultats sont regroupés
    dans un tableau.

    Paramètres
    ----------
    detector : str
        Code du détecteur, par exemple ``"V1"``.
    gps_start : float
        Temps GPS de début.
    gps_end : float
        Temps GPS de fin.

    Returns
    -------
    all_psd : numpy.ndarray
        PSD calculées pour chaque fenêtre temporelle.
    all_dates : numpy.ndarray
        Dates correspondant au début de chaque fenêtre.
    freq_ref : numpy.ndarray
        Fréquences associées aux PSD.
    """
    
    all_psd = []
    all_dates = []
    
    freq_ref = None
    window = 600
    n_windows = int(np.ceil((gps_end - gps_start) / window))
    current = gps_start

    for _ in tqdm(range(n_windows), desc="Téléchargement + PSD"):
        next_time = min(current + window, gps_end)
        try:
            strain = TimeSeries.fetch_open_data(
                detector,
                current,
                next_time,
                cache=True)

            freqs, psd = get_periodogram_strain(
                strain,
                plot=False)

            if freq_ref is None:
                freq_ref = freqs

            all_psd.append(psd)

        except Exception as e:

            print("\nPas de données :", current, e)
            if freq_ref is not None:
                all_psd.append(np.full(len(freq_ref), np.nan))

        all_dates.append(from_gps(current))
        current = next_time

    all_psd = np.array(all_psd)

    return (all_psd, np.array(all_dates), freq_ref)

def save_psd_strain(filename, all_psd, dates, freqs):
    np.savez(
        filename,
        all_psd=all_psd,
        dates=np.array(dates),
        freqs=freqs)

def load_psd_strain(filename):
    data = np.load(
        filename,
        allow_pickle=True)

    return (data["all_psd"], pd.to_datetime(data["dates"]), data["freqs"])

def spec_plot_strain(all_psd,dates,freqs,fmin=0,fmax=1000,title=None,log=True,outfile=None):
    """
    Représente un spectrogramme temporel de la PSD du strain.

    Paramètres
    ----------
    all_psd : numpy.ndarray
        PSD calculées pour les différentes fenêtres temporelles.
    dates : array-like
        Dates correspondant aux PSD.
    freqs : numpy.ndarray
        Fréquences associées aux PSD.
    fmin, fmax : float, optionnel
        Bornes fréquentielles affichées.
    title : str, optionnel
        Titre du graphique.
    log : bool, optionnel
        Si ``True``, utilise une échelle logarithmique pour les fréquences.
    outfile : str, optionnel
        Chemin du fichier dans lequel enregistrer la figure.
    """
    
    mask = ((freqs >= fmin) & (freqs <= fmax))

    plt.figure(figsize=(13,6))
    plt.pcolormesh(
        dates,
        freqs[mask],
        all_psd[:,mask].T,
        shading="auto")
    
    if log :
        plt.yscale("log")

    plt.xlabel("Date",size=15)
    plt.ylabel("Fréquences (Hz)",size=15)
    
    if title :
        plt.title(title,size=25)
    
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45,ha="right")
    
    plt.colorbar(label="PSD (dB)")
    
    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches="tight")

    plt.show()
    
def get_medians_strain(all_psd, freqs):
    y_list = []

    band_list = [(0,10),
        (10, 30),
        (30, 100),
        (100, 300),
        (300, 500)]

    for band in band_list:

        freq_band = ((freqs >= band[0]) & (freqs <= band[1]))

        median_psd = []

        for psd in all_psd:
            median_psd.append(np.median(psd[freq_band]))

        y_list.append(median_psd)

    return band_list,y_list
    
def median_4_strain(all_psd, all_dates, freqs,y=2024,limx=None, we=False, eq = False, p=80 ,n=5,outfile=None):
    if limx:
        a,b=limx
    else:
        a=all_dates[0]
        b=all_dates[-1]
        
    mask_xlim = (all_dates >= a) & (all_dates <= b)
    x = all_dates[mask_xlim]
 
    band_list,y_list = get_medians_strain(all_psd,  freqs)
    
    y1 = np.array(y_list[0])[mask_xlim]
    y2 = np.array(y_list[1])[mask_xlim]
    y3 = np.array(y_list[2])[mask_xlim]
    y4 = np.array(y_list[3])[mask_xlim]
    y5 = np.array(y_list[4])[mask_xlim]

    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(9, 8), sharex=True)

    axes[0].plot(x, y1, color='blue')
    axes[0].grid(True)
    axes[0].set_ylabel(f"{band_list[0][0]} - {band_list[0][1]} Hz")
    axes[0].set_xlim(a,b)

    axes[1].plot(x, y2, color='orange')
    axes[1].grid(True)
    axes[1].set_ylabel(f"{band_list[1][0]} - {band_list[1][1]} Hz")
    axes[1].set_xlim(a,b)


    axes[2].plot(x, y3, color='green')
    axes[2].grid(True)
    axes[2].set_ylabel(f"{band_list[2][0]} - {band_list[2][1]} Hz")
    axes[2].set_xlim(a,b)


    axes[3].plot(x, y4, color='red')
    axes[3].grid(True)
    axes[3].tick_params(axis="x", rotation=45)
    axes[3].set_ylabel(f"{band_list[3][0]} - {band_list[3][1]} Hz")
    axes[3].set_xlim(a,b)

    axes[4].plot(x, y5, color='green')
    axes[4].grid(True)
    axes[4].set_xlabel("Date")
    axes[4].tick_params(axis="x", rotation=45)
    axes[4].set_ylabel(f"{band_list[4][0]} - {band_list[4][1]} Hz")
    axes[4].set_xlim(a,b)
    
    ax = plt.gca()
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m\n%H:%M"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.tight_layout()
    
    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        
    plt.show()

#########################
######### METEO #########
#########################

def select_data(data,start, end, lat=43.6314, lon= 10.5045):
    local_data = data.sel(longitude = lon, latitude = lat, method = 'nearest')
    local_data_y = local_data.sel(valid_time = slice(start,end))
    dates_list =local_data_y['valid_time'].values 

    return local_data_y, dates_list

def get_wind_speed(wind_file):
    data_wind = xr.open_dataset(wind_file, engine='netcdf4')
    wind_u = data_wind["u10"]
    wind_v=data_wind["v10"]
    wind = np.sqrt(wind_u**2+wind_v**2)
    return wind

def plot_environment(file_temp, file_press, wind_file, rain_file, ocean_file, limx, p=80, n=5,outfile=None):
    """
    Représente les conditions environnementales autour du site de Virgo.

    Les séries temporelles de température, pression atmosphérique, vent,
    précipitations, vagues et houle sont extraites des fichiers NetCDF.
    Les périodes présentant des valeurs supérieures au percentile choisi
    sont mises en évidence.

    Paramètres
    ----------
    file_temp : str
        Fichier NetCDF contenant la température.
    file_press : str
        Fichier NetCDF contenant la pression atmosphérique.
    wind_file : str
        Fichier NetCDF contenant les composantes du vent.
    rain_file : str
        Fichier NetCDF contenant les précipitations.
    ocean_file : str
        Fichier NetCDF contenant les données de vagues et de houle.
    limx : tuple
        Intervalle temporel à représenter.
    p : float, optionnel
        Percentile utilisé pour définir les anomalies.
    n : int, optionnel
        Nombre minimal de valeurs consécutives au-dessus du seuil.
    outfile : str, optionnel
        Chemin du fichier dans lequel enregistrer la figure.
    """
    
    start, end = limx
    
    # =====================
    # TEMPERATURE + PRESSURE
    # =====================

    data_pressure = xr.open_dataset(file_press, engine='netcdf4')
    pressure = data_pressure["sp"]
    pressure_y, dates_pressure = select_data(pressure, start, end)

    data_temp = xr.open_dataset(file_temp, engine='netcdf4')
    temp = data_temp["t2m"]
    temp_y, dates_temp = select_data(temp, start, end)

    anomaly_temp = get_anomaly(dates_temp, temp_y.values, p, n)
    anomaly_pressure = get_anomaly(dates_pressure, pressure_y.values, p, n)

    # =====================
    # WIND + RAIN
    # =====================

    wind = get_wind_speed(wind_file)
    wind_y, dates_wind = select_data(wind, start, end)

    data_rain = xr.open_dataset(rain_file, engine='netcdf4')
    rain = data_rain["tp"]
    rain_y, dates_rain = select_data(rain*1000, start, end)

    anomaly_wind = get_anomaly(dates_wind, wind_y.values, p, n)
    anomaly_rain = get_anomaly(dates_rain, rain_y.values, p, n)

    # =====================
    # OCEAN
    # =====================

    data_ocean = xr.open_dataset(ocean_file, engine='netcdf4')

    wave = data_ocean["shww"]
    wave_y, dates_ocean = select_data(
        wave, start, end, lat=43.5, lon=10.3)

    swell = data_ocean["shts"]
    swell_y, dates_ocean = select_data(
        swell, start, end, lat=43.5, lon=10.3)

    anomaly_wave = get_anomaly(dates_ocean, wave_y.values, p, n)
    anomaly_swell = get_anomaly(dates_ocean, swell_y.values, p, n)

    # =====================
    # FIGURE
    # =====================

    fig, axes = plt.subplots(
        3, 1,
        figsize=(9.5, 12),
        sharex=True)

    # ---------------------
    # TEMPERATURE PRESSURE
    # ---------------------

    ax1 = axes[0]
    ax2 = ax1.twinx()

    ax1.plot(dates_temp, temp_y, color="coral", label="Température")
    ax2.plot(dates_pressure, pressure_y, color="darkturquoise", label="Pression")

    ax1.set_ylabel("Température (K)", color="coral",size=15)
    ax2.set_ylabel("Pression (Pa)", color="darkturquoise",size=15)

    ax1.tick_params(axis='y', labelcolor="coral")
    ax2.tick_params(axis='y', labelcolor="darkturquoise")

    for s,e in anomaly_temp:
        mask=(dates_temp>=s)&(dates_temp<=e)
        ax1.plot(np.array(dates_temp)[mask],
                 np.array(temp_y)[mask],
                 color="orangered",
                 linewidth=2)

    for s,e in anomaly_pressure:
        mask=(dates_pressure>=s)&(dates_pressure<=e)
        ax2.plot(np.array(dates_pressure)[mask],
                 np.array(pressure_y)[mask],
                 color="teal",
                 linewidth=2)

    ax1.set_title("Température et pression",size=15)

    # ---------------------
    # WIND RAIN
    # ---------------------

    ax3 = axes[1]
    ax4 = ax3.twinx()

    ax3.bar(dates_rain, rain_y,
            color="coral",
            alpha=0.5)

    ax4.plot(dates_wind, wind_y,
             color="darkturquoise")

    ax3.set_ylabel("Pluie/heure (mm)", color="coral",size=15)
    ax4.set_ylabel("Vitesse du vent (m/s)", color="darkturquoise",size=15)

    ax3.tick_params(axis='y', labelcolor="coral")
    ax4.tick_params(axis='y', labelcolor="darkturquoise")

    for s,e in anomaly_rain:
        mask=(dates_rain>=s)&(dates_rain<=e)
        ax3.bar(np.array(dates_rain)[mask],
                np.array(rain_y)[mask],
                color="orangered")

    for s,e in anomaly_wind:
        mask=(dates_wind>=s)&(dates_wind<=e)
        ax4.plot(np.array(dates_wind)[mask],
                 np.array(wind_y)[mask],
                 color="teal",
                 linewidth=2)

    ax3.set_title("Vitesse du vent et quantité de pluie par heure",size=15)

    # ---------------------
    # OCEAN
    # ---------------------

    ax5 = axes[2]
    ax6 = ax5.twinx()

    ax5.plot(dates_ocean,
             wave_y,
             color="coral")

    ax6.plot(dates_ocean,
             swell_y,
             color="darkturquoise")

    ax5.set_ylabel("Vague de vent (m)", color="coral",size=15)
    ax6.set_ylabel("Vague de houle (m)", color="darkturquoise",size=15)

    ax5.tick_params(axis='y', labelcolor="coral")
    ax6.tick_params(axis='y', labelcolor="darkturquoise")

    for s,e in anomaly_wave:
        mask=(dates_ocean>=s)&(dates_ocean<=e)
        ax5.plot(np.array(dates_ocean)[mask],
                 np.array(wave_y)[mask],
                 color="orangered",
                 linewidth=2)

    for s,e in anomaly_swell:
        mask=(dates_ocean>=s)&(dates_ocean<=e)
        ax6.plot(np.array(dates_ocean)[mask],
                 np.array(swell_y)[mask],
                 color="teal",
                 linewidth=2)

    ax5.set_title("Vagues océaniques",size=15)

    plt.setp(
        axes[-1].xaxis.get_majorticklabels(),
        rotation=45)

    fig.suptitle(
        "Conditions environnementales près du site de Virgo",
        fontsize=25)

    fig.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches="tight")

    plt.show()