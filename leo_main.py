import data_scheduler.lib_data_merger as data_merger

if __name__ == '__main__':
    mice_data_dir = r'C:\Users\lkokot\Desktop\ETHZ_STAT_MSC\sem_2\stats_lab\analysis\CSV data files for analysis'
    md = data_merger.MiceDataMerger(mice_data_dir)
    eth = md.fetch_mouse_data(167, 'eth')
    # eth = md.fetch_mouse_signal(167, 'eth', 'rq')
    print(eth['v_o2'].get_pandas())

    m_167 = md.fetch_mouse_data(167)
    print(list(m_167))

    mice_data = md.fetch_mouse_data([167, 2934, 165])
    print(list(mice_data))
    print(mice_data)

    mice_data = md.fetch_mouse_data([306], signals = 'brain_signal')
    print(list(mice_data))
    print(mice_data)