# START HERE TO USE GOVT TARGETS
import numpy as np
import toolkit as kit
import torch
import sys
import argparse



parser = argparse.ArgumentParser(description='Inference on government data')
parser.add_argument('--scenario', help='Scenario name')
parser.add_argument('--models-dir', type=str, default='/home/soap/software/my_SOAP_clean/SOAP/training_128_subbands/separable_model/', help='Path to saved models directory')
parser.add_argument('--model', type=str, default='beam_formed_4x33', help='Model option')
parser.add_argument('--num-pulses', type=int, default=32, help='Number of pulses in CPI')
parser.add_argument('--sampling-rate', type=float, default=1.25e9, help='Sampling rate in Hz')
parser.add_argument('--pulse-duration', type=float, default=8e-6, help='Pulse duration in seconds')
parser.add_argument('--pri-duration', type=float, default=80e-6, help='PRI duration in seconds')
parser.add_argument('--bandwidth', type=float, default=520e6, help='Bandwidth in Hz')
parser.add_argument('--center-frequency', type=float, default=4e9, help='Center frequency in Hz')
parser.add_argument('--num-bands', type=int, default=128, help='Number of frequency bands')
parser.add_argument('--data-path', type=str, default='./', help='Path to .npy data files')
parser.add_argument('--ref-code-path', type=str, default='./RefArchDetectionCode', help='Path to RefArchDetectionCode')

args = parser.parse_args()

num_pulses = int(args.num_pulses)
sampling_rate = float(args.sampling_rate)
pulse_duration = float(args.pulse_duration)
pri_duration = float(args.pri_duration)
bandwidth = float(args.bandwidth)
center_frequency = float(args.center_frequency)
ref_code_path = args.ref_code_path
data_path = args.data_path
num_bands = int(args.num_bands)
scenario = args.scenario
model_option = args.model # number of pulses in CPI
saved_model_dir = args.models_dir

pri_length = int(pri_duration * sampling_rate)
print(f"PRI length in samples: {pri_length}")
print(num_bands)
print(pri_length/num_bands)

#model setup
# saved_model_dir = "/home/soap/software/my_SOAP_clean/SOAP/training_128_subbands/separable_model/"

print(f"Using model option: {model_option}")
if model_option == "4x13":
    import model as model 
    saved_model_weights = "model_4x32_radar_coords_scaleby3_65"
    model_timesteps = 256  # for 4x32 models

elif model_option == "4x33":
    import LISTA_model_4x32 as model
    #np.round
    # saved_model_weights = "model_4x32_radar_coords_scaleby5_65/"
    #adc
    saved_model_weights = "model_4x32_radar_adcscaledby1_65/"
    model_timesteps = 256  # for 4x32 models

elif model_option == "beam_formed_4x33":
    import LISTA_model_4x32 as model
    #adc
    saved_model_dir = "/home/soap/software/SOAP/separable_model/"
    saved_model_weights = "model_4x33_6scenarios_beamformed_true_values_68/"
    model_timesteps = 256  # for 4x32 models

elif model_option == "4x33_scaledby0.15": 
    import LISTA_model_4x32 as model
    saved_model_weights = "model_4x33_window_A2C_4_pts_per_degree_256scaledby0.15_65/"
    model_timesteps = 256  # for 4x32 models
elif model_option == "4x33_20_targets_scaledby0.15": 
    import LISTA_model_4x32 as model
    saved_model_weights = "model_4x33_window_A2C_4_pts_per_degree_20_targets_256scaledby0.15_65/"
    model_timesteps = 256  # for 4x32 models

elif model_option == "4x33_20_targets_scaledby1": 
    import LISTA_model_4x32 as model
    saved_model_weights = "model_4x33_window_A2C_4_pts_per_degree_20_targets_256scaledby1.0_65/"
    model_timesteps = 256  # for 4x32 models
    
elif model_option == "4x25_dilation":
    import LISTA_model_4x25_dilation as model
    saved_model_weights = "model_4x25_dilation_window_512scaledby0.15_65/"
    model_timesteps = 512  # for 4x32 models
elif model_option ==  "mvdr":
    model_timesteps = pri_length/num_bands
    print(model_timesteps)
elif model_option == "f32":
    import LISTA_model_4x32 as model
    saved_model_weights = "true_data_f32/"
    model_timesteps = 256
elif model_option == "bf16":
    import LISTA_model_4x32 as model
    saved_model_weights = "true_data_bf16/"
    model_timesteps = 256
    pass
else:
    import LISTA_model_4x25 as model
    saved_model_weights = "model_4x25_window_512scaledby0.15_65/"
    model_timesteps = 512



#set up separable model
if model_option != "mvdr":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.LISTA_Net()
    effective_path = saved_model_dir + saved_model_weights+ "beamforming_model_final.pt"
    model.load_state_dict(torch.load(effective_path)["model_state_dict"])
    model.to(device)
    model.eval()



# needs to be time-flipped for convolution
pulse_replica = kit.generate_LFM_pulse(sampling_rate, pulse_duration, bandwidth, time_flip=True)

# use "sample_interferers.py" to create this file given scenario data
data = np.load(f"{scenario}/{scenario}_interf.npy")
# data = np.load(f"{scenario}/{scenario}_nointerf.npy")

print(data.shape)
print(np.min(np.abs(data))
      , np.max(np.abs(data))
        )

data_scenario_a1 = np.load(f"A1/A1_interf.npy")
print(data_scenario_a1.shape)
print(np.min(np.abs(data_scenario_a1))
      , np.max(np.abs(data_scenario_a1))
        )


angs = np.load(f"{scenario}/{scenario}_targets_ang.npy")
poss = np.load(f"{scenario}/{scenario}_targets_pos.npy")
vels = np.load(f"{scenario}/{scenario}_targets_vel.npy") - np.asarray([0, 90, 0])

ranges = np.sqrt(np.sum(poss**2, axis=1))
radial_vels = (poss * vels).sum(axis=1) / ranges


pos = kit.create_shifted_positions(f"{scenario}/{scenario}_element_positions.npy")

csv_buffer = "Idx, Actual_Range, Actual_Doppler, Recovd_Range, Recovd_Doppler, Recovd_SNR\n"


    
for target_idx in range(len(ranges)):
    az = np.deg2rad(angs[target_idx, 0]) 
    el = np.deg2rad(angs[target_idx, 1])

    # print("Azimuth: ", angs[target_idx, 0], " Elevation: ", angs[target_idx, 1])
    # print("Range: ", ranges[target_idx])
    # print("Velocity: ", radial_vels[target_idx])

    # change this to beamform_ml_polyphase_2D to use ML beamforming
    if model_option == "mvdr":
        bf_out = kit.beamform_mvdr_polyphase_2D(data.reshape(128, -1)[:, :], num_bands, num_pulses, pos, az, el, pri_length, sampling_rate, center_frequency)
    else:
        bf_out = kit.beamform_ml_polyphase_2D(data.reshape(128, -1)[:, :], num_bands, num_pulses, pos, az, el, pri_length, sampling_rate, center_frequency,model, timesteps=256)

    data_cube = bf_out

    print(f"{target_idx} Actual: {ranges[target_idx]:.2f}, {radial_vels[target_idx]:.2f}")
    print(center_frequency, sampling_rate, pri_duration)
    dist, vel, snr, reports = kit.run_detection_pipeline_single(data_cube, pulse_replica, "./RefArchDetectionCode", sampling_rate, pri_duration, center_frequency, pc_taper=True, simulated_data=False, radial_vel=None, viz=True, vmax=0.1, target_idx=target_idx)
    # print(reports["range"])

    target_csv_buffer = "Idx, Actual_Range, Actual_Doppler, Recovd_Range, Recovd_Doppler, Recovd_SNR\n"
    print("Detections:{}".format(len(reports["Doppler"])))
    for i in range(len(reports["Doppler"])):
        correct_range = np.abs(reports["range"][i] - ranges[target_idx]) < 20
        correct_vel = np.abs(reports["Doppler"][i] - radial_vels[target_idx]) < 20
        target_csv_buffer += f'{target_idx}, {ranges[target_idx]}, {radial_vels[target_idx]}, {reports["range"][i][0]}, {reports["Doppler"][i][0]}, {reports["sinr"][i][0]}\n'
        
        if correct_range and correct_vel:
            print(f"{target_idx} Recovd: {reports['range'][i][0]:.2f}, {reports['Doppler'][i][0]:.2f}, {reports['sinr'][i][0]:.2f}")
            csv_buffer += f'CORRECT, {target_idx}, {ranges[target_idx]}, {radial_vels[target_idx]}, {reports["range"][i][0]}, {reports["Doppler"][i][0]}, {reports["sinr"][i][0]}\n'
            break
    
    else:
        print("Target not found :(")
        if(len(reports['range'])==0):
            print("No detections at all")
            csv_buffer += f'NO_DETECTIONS, {target_idx}, {ranges[target_idx]}, {radial_vels[target_idx]}, N/A, N/A, N/A\n'
            print("")
            continue
        closest_idx = np.argmin(np.abs(reports['range'] - ranges[target_idx]))
        print(f"target with closest range: {reports['range'][closest_idx][0]} idx = {closest_idx}  ")
        csv_buffer += f"CLOSEST,{target_idx}, {ranges[target_idx]}, {radial_vels[target_idx]}, {reports['range'][closest_idx][0]},{reports['Doppler'][closest_idx][0]}, {reports['sinr'][closest_idx][0]}\n"
    print("")
    
    with open(f"all_detections_model_{model_option}_target_{target_idx}_{scenario}_res.csv", "w") as target_handle:
        target_handle.write(target_csv_buffer)
    



with open(f"all_detections_model_{model_option}_{model_timesteps}_{scenario}_res.csv", "w") as handle:
    handle.write(csv_buffer)
    
print(csv_buffer)
