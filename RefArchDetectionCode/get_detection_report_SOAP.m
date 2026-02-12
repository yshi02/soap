function detection_report = get_detection_report_SOAP(CPI_data, sigpro_params)

% get_detection_report_SOAP() performs the detection and estimation functions in 
% radar signal processing chain. This function is a simplified version of the
% one used in Reference Architecture, but functionally all operations are identical.

% get_detection_report_SOAP() calls functions mean_level_cfar() and cluster() 

% Input:
%   CPI_data - struct containing pule-compressed, Doppler-processed data  
%       .num_Doppler_bins    - number of Doppler ins (32 in Reference Architecture)     
%       .range_cell_width    - width of range cell (0.24 m in Reference Architecture)
%       .range_bins          - vector of range bin values
%       .Doppler_bins        - vector of Doppler bin values
%       .data_cube           - 2-D datacube of pulse-compressed and Doppler
%                              processed data. Ensure that its dimensions are 
%                              range x Doppler
%   sigpro_params - signal processing parameters
%       .CFAR_threshold      - CFAR threshold value (10 dB in Reference Architecture)
%       .num_of_cells        - number of cells in CFAR (100 in Reference Architecture)
%       .num_of_holes        - number of holes in CFAR (5 in Reference Architecture)

% Output:
%   detection_report - struct containing estimated parameters for all detections
%       .range        - range estimates of detections
%       .Doppler      - Doppler estimates of detections
%       .sinr         - SINR estimates of detections

% Authors: Minhtri Ho and Hedi Krichene
% Organization: The Johns Hopkins University/Applied Physics Laboratory
% Baseline date: 10-28-13

%% Unpack input parameter structs to create local variables

field_names = fieldnames(sigpro_params);
for field_ind = 1:length(field_names)
    eval(strcat(field_names{field_ind}, ' = sigpro_params.', field_names{field_ind}, ';'));
end
clear sigpro_params;

%% CFAR detection

% Compute the signal power
local_data_cube = CPI_data.data_cube;
power_matrix = abs(local_data_cube).^2; % Matrix is in Range x Doppler order

% Perform mean-level CFAR detection
[rddet, cellave] = mean_level_cfar(power_matrix, CFAR_threshold, num_of_cells, num_of_holes);

% rddet is now the detection flag matrix with detections clustered at
% adjacent bins.  So, there is the need to find the target detection
% among the clustered detections.

% Call the cluster function
detection_matrix = cluster(power_matrix,rddet);

% Retrieve the indices of the detections in the detection matrix
idet = find(detection_matrix); % One-dimensional indices
[idet_row, idet_column] = find(detection_matrix); % Two-dimensional indices
num_of_detections = length(idet); % Number of detections

%% Range estimation

% Estimation of Range (using three-point interpolation)
range_estimate = zeros(num_of_detections,1); % Prepare array
range_index = zeros(num_of_detections, 1);
delta_range = CPI_data.range_cell_width;
for detection_ind = 1:num_of_detections % Loop over all the detections (of this target beam)
    range_index(detection_ind) = idet_row(detection_ind);
    X0 = CPI_data.range_bins(idet_row(detection_ind)); % Range bin value
    if idet_row(detection_ind) ~= 1 && idet_row(detection_ind) ~= length(CPI_data.range_bins)
        % If the detection range bin does not fall on the first or last bin
        P0 = power_matrix(idet_row(detection_ind), idet_column(detection_ind)); % Power
        Pl = power_matrix(idet_row(detection_ind)-1, idet_column(detection_ind)); % Power on left
        Pr = power_matrix(idet_row(detection_ind)+1, idet_column(detection_ind)); % Power on right
        range_estimate(detection_ind) = X0+0.5*delta_range*(Pr-Pl)/(2*P0-Pr-Pl); % 3-point algorithm
    else
        % If the detection range bin falls on the first or last bin, then
        % we do not do the interpolation but just take the value of the
        % range bin.
        range_estimate(detection_ind) = X0;
    end
end

%% Doppler estimation

% Estimation of Doppler (using three-point interpolation)
doppler_estimate = zeros(num_of_detections, 1);
doppler_index    = zeros(num_of_detections, 1);
delta_Doppler = CPI_data.Doppler_bins(2)-CPI_data.Doppler_bins(1);
% Loop over all detections in the detection beam
for detection_ind = 1:num_of_detections
    doppler_index(detection_ind) = idet_column(detection_ind);
    X0 = CPI_data.Doppler_bins(idet_column(detection_ind)); % Doppler bin value
    P0 = power_matrix(idet_row(detection_ind),idet_column(detection_ind)); % Power
    % If detection Doppler bin falls on the first bin
    if idet_column(detection_ind) == 1 
        Pl = power_matrix(idet_row(detection_ind), CPI_data.num_Doppler_bins); % Wrap around on left
        Pr = power_matrix(idet_row(detection_ind), idet_column(detection_ind)+1); % As usual on right
    elseif idet_column(detection_ind) == CPI_data.num_pulse_Doppler % If falls on the last bin
        Pl = power_matrix(idet_row(detection_ind), idet_column(detection_ind)-1); % As usual on left
        Pr = power_matrix(idet_row(detection_ind), 1); % Wrap around on right
    else % Nothing special, just the usual bins in the middle
        Pl = power_matrix(idet_row(detection_ind), idet_column(detection_ind)-1); % Power on left
        Pr = power_matrix(idet_row(detection_ind), idet_column(detection_ind)+1); % Power on right
    end
    doppler_estimate(detection_ind) = X0+0.5*delta_Doppler*(Pr-Pl)/(2*P0-Pr-Pl); % 3-point algorithm
end

%% Populate detection_report struct

detection_report.range   = range_estimate;
detection_report.Doppler = doppler_estimate;
detection_report.sinr    = (10*log10(power_matrix(idet))-cellave(idet));
