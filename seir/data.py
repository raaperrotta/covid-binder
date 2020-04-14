import pandas as pd


POPULATION_OF_LOMBARDY = 10_060_574
DATE_OF_LOMBARDY_RED_ZONES = pd.to_datetime('22 Feb 2020')
DATE_OF_LOMBARDY_LOCKDOWN = pd.to_datetime('8 Mar 2020')
DATE_OF_SHUTDOWN_OF_NONESSENTIALS = pd.to_datetime('21 Mar 2020')

regioni = pd.read_csv(
    'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv',
    parse_dates=['data']
)
lombardia = regioni[regioni['denominazione_regione'] == 'Lombardia'].copy()
lombardia['data'] = pd.to_datetime(lombardia['data'].dt.date)  # Drop the time
lombardia.sort_values('data', inplace=True)
