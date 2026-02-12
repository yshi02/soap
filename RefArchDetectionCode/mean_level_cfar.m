function [range_Doppler_detections, cell_averages] = mean_level_cfar(power_matrix, CFAR_threshold, num_of_cells, num_of_holes)

% mean_level_cfar() used the CFAR algorithm to compute the average noise power 
% along the range dimension only; then thresholds these averages to compute
% detections.

% Input: 
%   power_matrix  - power (squared magnitude) matrix
%   threshold     - threshold value (dB)
%   num_of_cells  - total number of averaging cells; must be even 
%   num_of_holes  - number of hole cells on each side

% Output: 
%   range_Doppler_detections  - Detection flag matrix consisting of 1's and 0's
%   cell_averages  - Computed average noise+inteference power (dB)

% Authors: Minhtri Ho and Hedi Krichene
% Organization: The Johns Hopkins University/Applied Physics Laboratory
% Baseline date: 10-28-13

%% Compute cell average for each pixel

% Compute number of range bins
[num_of_range_bins, junk] = size(power_matrix) ; 
cell_averages = zeros(size(power_matrix));
% Compute number of averaging cells on each side
num_of_cells_side = num_of_cells/2 ; 

% Loop over range bins (for sliding the averaging window) 
for range_ind = 1:num_of_range_bins   
    if range_ind <= (num_of_holes+1)
        % This is the case where there are NO averaging cells on the LOWER
        % side.  Therefore all the averaging is done for the upper side
        range_ind_cells_upper = (range_ind+num_of_holes+1):(range_ind+num_of_holes+2*num_of_cells_side);
        range_ind_cells = range_ind_cells_upper;
    elseif range_ind > (num_of_holes+1) && range_ind <= (num_of_holes+num_of_cells_side) 
        % This is the case where there are SOME (but NOT ENOUGH) averaging 
        % cells on the LOWER side.  Therefore the averaging cells are 
        % compensated on the upper side
        range_ind_cells_lower = 1:(range_ind - num_of_holes - 1);
        number_cells_left = num_of_cells_side - length(range_ind_cells_lower);
        range_ind_cells_upper = (range_ind + num_of_holes + 1):(range_ind + num_of_holes + num_of_cells_side + number_cells_left);
        range_ind_cells = [range_ind_cells_lower,range_ind_cells_upper];
    elseif range_ind >= (num_of_cells_side + num_of_holes + 1) && range_ind <= (num_of_range_bins - num_of_cells_side - num_of_holes) 
        % This is the case where there are ENOUGH required number of
        % averaging cells on BOTH sides
        range_ind_cells_lower = (range_ind - num_of_holes - num_of_cells_side):(range_ind - num_of_holes - 1);
        range_ind_cells_upper = (range_ind+num_of_holes + 1):(range_ind + num_of_holes + num_of_cells_side);
        range_ind_cells = [range_ind_cells_lower,range_ind_cells_upper];
    elseif range_ind > (num_of_range_bins - num_of_holes - num_of_cells_side) && range_ind <= (num_of_range_bins - num_of_holes - 1)
        % This is the case where there are SOME (but not ENOUGH) averaging
        % cells on the UPPER side.  Therefore the averaging cells are
        % compensated on the lower side
        range_ind_cells_upper = (range_ind + num_of_holes + 1):num_of_range_bins;
        number_cells_left = num_of_cells_side-length(range_ind_cells_upper);
        range_ind_cells_lower = (range_ind - num_of_holes - num_of_cells_side - number_cells_left):(range_ind - num_of_holes - 1);
        range_ind_cells = [range_ind_cells_lower, range_ind_cells_upper];
    else
        % This is the case where there are NO averaging cells on the UPPER
        % side.  Therefore all the averaging is done for the lower side
        range_ind_cells_lower = (range_ind - num_of_holes - 2*num_of_cells_side):(range_ind - num_of_holes - 1);
        range_ind_cells = range_ind_cells_lower;
    end
    
    % Computer cell average
    cell_averages(range_ind,:) = sum(power_matrix(range_ind_cells,:))/num_of_cells;
 
end

%% Threshold the cell averages

% Convert cellave to dB
cell_averages = 10*log10(cell_averages);
power_matrix = 10*log10(power_matrix);

% Also, convert cell power to dB

% Test the signal power with the average noise to return the detection
% matrix
range_Doppler_detections = (power_matrix - cell_averages) > CFAR_threshold;