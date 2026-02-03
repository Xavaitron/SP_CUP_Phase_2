function generate_rir_data()
    %% GENERATE_RIR_DATA - Generates RIR and spatial data for SP Cup submission
    %
    %  Creates rir_data.mat containing Room Impulse Responses for both
    %  anechoic and reverberant conditions, used by prepare_submission.m
    %
    %  Output: rir_data.mat with:
    %    - anechoic.rir_target, anechoic.rir_interf, anechoic.params
    %    - reverb.rir_target, reverb.rir_interf, reverb.params
    
    clc; close all;
    fprintf('======================================\n');
    fprintf('  RIR DATA GENERATOR FOR SUBMISSION\n');
    fprintf('======================================\n\n');
    
    %% CONFIGURATION (matching dataset generation scripts)
    % Room parameters
    roomDimensions = [4.9, 4.9, 4.9];  % meters
    
    % Microphone array (2-mic, 8cm spacing, center of room)
    micPositions = [2.41, 2.45, 1.5;   % Mic 1
                    2.49, 2.45, 1.5];  % Mic 2
    micCenter = mean(micPositions, 1);
    arraySpacing = 0.08;  % 8cm
    
    % Source geometry
    sourceRadius = 1.0;   % meters from mic center
    sourceHeight = 1.5;   % meters
    
    % Fixed angles for submission (single target, single interferer)
    targetAngle = 90;     % Target at 90 degrees (front)
    interfAngle = 40;     % Interferer at 40 degrees
    
    % Audio parameters
    fs = 16000;           % Sample rate
    c = 343;              % Speed of sound (m/s)
    n_rir = 4096;         % RIR length (samples)
    
    % SNR/SIR parameters
    snr_dB = 5;
    sir_dB = 0;
    
    %% CHECK RIR GENERATOR
    fprintf('1. Checking rir_generator MEX function...\n');
    
    % Add RIR_gen to path if needed
    rirGenPath = fullfile(pwd, 'RIR_gen');
    if exist(rirGenPath, 'dir')
        addpath(rirGenPath);
        fprintf('   - Added RIR_gen to path\n');
    end
    
    if ~exist('rir_generator', 'file')
        error(['rir_generator not found!\n' ...
               'Please compile it first:\n' ...
               '  cd RIR_gen\n' ...
               '  mex -setup\n' ...
               '  mex rir_generator.cpp rir_generator_core.cpp']);
    end
    fprintf('   - rir_generator found\n');
    
    %% GENERATE ANECHOIC RIRs (RT60 = 0)
    fprintf('\n2. Generating ANECHOIC RIRs (RT60 = 0)...\n');
    
    rt60_anechoic = 0.0;
    anechoic = generateRIRSet(c, fs, micPositions, micCenter, roomDimensions, ...
                               rt60_anechoic, n_rir, sourceRadius, sourceHeight, ...
                               targetAngle, interfAngle, snr_dB, sir_dB, arraySpacing);
    fprintf('   - Target RIR: [%d x %d]\n', size(anechoic.rir_target));
    fprintf('   - Interf RIR: [%d x %d]\n', size(anechoic.rir_interf));
    
    %% GENERATE REVERBERANT RIRs (RT60 = 0.5)
    fprintf('\n3. Generating REVERBERANT RIRs (RT60 = 0.5)...\n');
    
    rt60_reverb = 0.5;
    reverb = generateRIRSet(c, fs, micPositions, micCenter, roomDimensions, ...
                             rt60_reverb, n_rir, sourceRadius, sourceHeight, ...
                             targetAngle, interfAngle, snr_dB, sir_dB, arraySpacing);
    fprintf('   - Target RIR: [%d x %d]\n', size(reverb.rir_target));
    fprintf('   - Interf RIR: [%d x %d]\n', size(reverb.rir_interf));
    
    %% SAVE TO FILE
    fprintf('\n4. Saving rir_data.mat...\n');
    
    outputFile = 'rir_data.mat';
    save(outputFile, 'anechoic', 'reverb', '-v7.3');
    
    fprintf('   - Saved: %s\n', fullfile(pwd, outputFile));
    
    %% SUMMARY
    fprintf('\n======================================\n');
    fprintf('  RIR DATA GENERATION COMPLETE!\n');
    fprintf('======================================\n');
    fprintf('\nContents of rir_data.mat:\n');
    fprintf('  anechoic: rir_target [%d x %d], rir_interf [%d x %d], RT60=%.1f\n', ...
            size(anechoic.rir_target), size(anechoic.rir_interf), rt60_anechoic);
    fprintf('  reverb:   rir_target [%d x %d], rir_interf [%d x %d], RT60=%.1f\n', ...
            size(reverb.rir_target), size(reverb.rir_interf), rt60_reverb);
    fprintf('\nNext step: Run prepare_submission to generate submission folder.\n');
end


%% ========================================================================
function data = generateRIRSet(c, fs, micPos, micCenter, roomDim, rt60, n_rir, ...
                                 radius, height, targetAngle, interfAngle, snr, sir, spacing)
    % Generate RIRs for one condition (anechoic or reverb)
    % Uses rir_generator for both - RT60=0 for anechoic, RT60=0.5 for reverb
    
    % Calculate source positions
    targetPos = getPositionFromAngle(targetAngle, radius, micCenter, height);
    interfPos = getPositionFromAngle(interfAngle, radius, micCenter, height);
    
    % Generate RIRs using rir_generator
    % rir_generator returns [n_mics x n_samples], transpose to [n_samples x n_mics]
    rir_t = rir_generator(c, fs, micPos, targetPos, roomDim, rt60, n_rir);
    rir_i = rir_generator(c, fs, micPos, interfPos, roomDim, rt60, n_rir);
    data.rir_target = rir_t';  % [4096 x 2]
    data.rir_interf = rir_i';  % [4096 x 2]
    
    % Build params struct
    data.params.sampling_rate = fs;
    data.params.mic_positions = micPos;
    data.params.array_spacing = spacing;
    data.params.source_azimuth = targetAngle;
    data.params.interferer_azimuth = interfAngle;
    data.params.source_height = height;
    data.params.source_radius = radius;
    data.params.SNR_dB = snr;
    data.params.SIR_dB = sir;
    data.params.RT60 = rt60;
    data.params.room_dimensions = roomDim;
    data.params.n_rir = n_rir;
end


%% ========================================================================
function pos = getPositionFromAngle(angle_deg, radius, center, height)
    % Convert azimuth angle to 3D position
    % 0° = right, 90° = front, 180° = left
    rad = deg2rad(angle_deg);
    pos = [center(1) + radius * cos(rad), ...
           center(2) + radius * sin(rad), ...
           height];
end
