# Seismic Noise Analysis Around the Virgo Gravitational-Wave Detector

This repository contains the Python code developed during my internship
on the analysis of seismic noise around the Virgo gravitational-wave detector.

The project investigates the temporal and spectral variability of seismic
noise near Virgo and explores possible relationships with environmental
conditions, earthquakes, detector sensitivity and gravitational-wave data.

## Objectives

The main objectives of the project are:

- characterize seismic noise recorded near the Virgo detector;
- compute hourly power spectral densities (PSD);
- study the temporal evolution of seismic noise in different frequency bands;
- identify periods of unusually high seismic noise;
- investigate local, regional and global earthquake activity;
- study environmental parameters such as wind, precipitation, temperature,
  atmospheric pressure and ocean waves;
- analyse publicly available Virgo data from GWOSC;
- investigate detector sensitivity indicators such as the BNS range and duty cycle;
- analyse publicly available strain data and their spectral properties.

## Data

The project uses several sources of publicly available data. Raw and processed data are not included in this repository because some
datasets can be large. The repository contains the code required to download and process the
data.


### Seismic data

Seismic data are obtained through the FDSN web services using
[ObsPy](https://docs.obspy.org/).

The main seismic station used in the analysis is:

- Network: `IV`
- Station: `PII`
- Channel: `HHZ`

Data are downloaded in MiniSEED format and must be corrected
for the instrumental response.

### Gravitational-wave data

Gravitational-wave event and detector data are obtained from the
[Gravitational-Wave Open Science Center (GWOSC)](https://gwosc.org/).

The analysis includes data from observing runs such as:

- O2
- O3a
- O3b
- O4b

### Environmental data

Meteorological and oceanographic data are obtained from ERA5/Copernicus
datasets and processed using `xarray`.

The analysed parameters include:

- temperature;
- atmospheric pressure;
- wind speed;
- precipitation;
- wind waves;
- ocean swell.


## Author

Clémence Georges

Master 1 Geophysics and Earth Imaging
Université Grenoble Alpes

# Internship

This project was developed as part of a research internship on the
analysis of seismic noise around gravitational-wave detectors.

Supervisors: Jean Soubestre & Pierre Boué
