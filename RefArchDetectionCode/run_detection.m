function [range, velocity, snr, report] = run_detection(doppler_cube_filename)

load(doppler_cube_filename);

CPI_data = struct();
CPI_data.num_Doppler_bins = num_Doppler_bins;
CPI_data.range_cell_width = range_cell_width;
CPI_data.range_bins = range_bins;
CPI_data.Doppler_bins = Doppler_bins;
CPI_data.data_cube = data_cube;
CPI_data.num_pulse_Doppler = 32;

sigpro_params = struct();
sigpro_params.CFAR_threshold = 10;
sigpro_params.num_of_cells = 100;
sigpro_params.num_of_holes = 5;

report = get_detection_report_SOAP(CPI_data, sigpro_params);

% hack: get index of max SNR. This may not be correct.
[val, idx] = max(report.sinr);

range = report.range(idx);
velocity = report.Doppler(idx);
snr = val;
