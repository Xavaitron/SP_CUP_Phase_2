% This script generates anechoic test dataset samples
% with fixed target and interferer angles.

function test_anechoic()
    clc; close all;


    %% 1. CONFIGURATION
    
    maleDir   = "D:\Dataset\Male";           
    femaleDir = "D:\Dataset\Female";         
    noiseDir  = "C:\Users\ironp\Downloads\SP Cup\matlab_stuff\MUSAN\noise";    
    musicDir  = "C:\Users\ironp\Downloads\SP Cup\matlab_stuff\MUSAN\music";    
    
    % Changed output folder name to avoid overwriting your previous data
    outputDatasetDir = '../Test_Dataset/anechoic';
    
    % --- Generation Settings ---
    numSamplesToGenerate = 5000;
    desiredLength_s      = 4.0;
    fs                   = 16000;
    desiredLength_samples = floor(desiredLength_s * fs);
    
    % --- Acoustic Settings ---
    sir_dB            = 0;     % Signal-to-Interference Ratio
    snr_dB            = 5;     % Signal-to-Noise Ratio
    source_radius_m   = 1.0;
    roomDimensions    = [4.9, 4.9, 4.9];
    
    % --- FIXED ANGLES ---
    targetAngle_deg   = 90;
    interfererAngle_deg = 40;
    
    % Mic Array (2 Mics)
    micPositions      = [2.41, 2.45, 1.5;   
                         2.49, 2.45, 1.5];
    micCenter         = mean(micPositions, 1);
    
    c     = 343;   % Speed of sound
    beta  = 0.0;   % RT60 (Anechoic)
    n_rir = 4096;  % RIR Length


    if ~exist(outputDatasetDir, 'dir'), mkdir(outputDatasetDir); end


    %% 2. FILE SCANNING & MEMORY OPTIMIZATION
    fprintf('1. Scanning Audio Files...\n');
    
    rawMale   = getFiles(maleDir);
    rawFemale = getFiles(femaleDir);
    rawNoise  = getFiles(noiseDir);
    rawMusic  = getFiles(musicDir);


    if isempty(rawMale), error('No Male files found! Check paths.'); end


    % --- Store file lists in shared memory ---
    fprintf('   - Storing file lists in shared memory...\n');
    cMale   = parallel.pool.Constant(rawMale);
    cFemale = parallel.pool.Constant(rawFemale);
    cNoise  = parallel.pool.Constant(rawNoise);
    cMusic  = parallel.pool.Constant(rawMusic);


    %% 3. SAFE PARALLEL POOL SETUP
    SAFE_WORKERS = 4; 
    poolObj = gcp('nocreate');
    if isempty(poolObj)
        parpool('local', SAFE_WORKERS);
    elseif poolObj.NumWorkers > SAFE_WORKERS
        delete(poolObj);
        parpool('local', SAFE_WORKERS);
    end


    %% 4. RIR PRE-CALCULATION (Specific for Fixed Angles)
    fprintf('2. Pre-calculating specific RIRs for 90 and 40 degrees...\n');
    
    % Calculate positions
    posT = getPositionFromAngle(targetAngle_deg, source_radius_m, micCenter);
    posI = getPositionFromAngle(interfererAngle_deg, source_radius_m, micCenter);
    
    % Generate just the two needed RIRs
    rirT_raw = rir_generator(c, fs, micPositions, posT, roomDimensions, beta, n_rir);
    rirI_raw = rir_generator(c, fs, micPositions, posI, roomDimensions, beta, n_rir);
    
    % Store in Constant memory so all workers can access them instantly
    cRirTarget = parallel.pool.Constant(rirT_raw);
    cRirInterf = parallel.pool.Constant(rirI_raw);
    
    fprintf('   - RIRs generated and stored.\n');


    %% 5. MAIN GENERATION LOOP
    fprintf('3. Starting Dataset Generation (Target: 90, Interf: 40)...\n');
    
    % Progress Queue
    q = parallel.pool.DataQueue;
    afterEach(q, @(x) updateProgress(x, numSamplesToGenerate));
    
    parfor i = 1:numSamplesToGenerate
        rng('shuffle'); % Ensure random variety
        
        try
            % --- A. Source Selection ---
            % Target is always Male
            mList = cMale.Value;
            targetFile = mList{randi(numel(mList))};
            sourceType = "Male"; % Set source label
            
            % Interferer Distribution: 33% Female, 33% Music, 33% Noise
            roll = rand();
            interfType = ""; % Initialize string
            
            if roll < (1/3)
                iList = cFemale.Value; % 0 to 0.33
                interfType = "Female";
            elseif roll < (2/3)
                iList = cMusic.Value;  % 0.33 to 0.66
                interfType = "Music";
            else
                iList = cNoise.Value;  % 0.66 to 1.0
                interfType = "Noise";
            end
            
            % Fallback if list is empty
            if isempty(iList) 
                iList = cNoise.Value; 
                interfType = "Noise";
            end
            interfererFile = iList{randi(numel(iList))};
            
            % Process Audio
            targetSignal = preprocessAudio(targetFile, fs, desiredLength_samples);
            interfererSignal = preprocessAudio(interfererFile, fs, desiredLength_samples);
            
            % --- B. RIR Retrieval (Fixed) ---
            % No calculation needed, just pull from constant memory
            rirT = cRirTarget.Value;
            rirI = cRirInterf.Value;
            
            % Convolution
            micT = fftfilt(rirT', targetSignal);
            micI = fftfilt(rirI', interfererSignal);
            
            % Truncate to match length
            len = min(size(micT, 1), size(micI, 1));
            micT = micT(1:len, :);
            micI = micI(1:len, :);
            
            % --- C. Mixing (SIR & SNR) ---
            powT = mean(micT(:,1).^2) + 1e-12;
            powI = mean(micI(:,1).^2) + 1e-12;
            
            % Adjust Interferer to satisfy SIR = 0dB
            scale = sqrt(powT / powI) / (10^(sir_dB/20));
            micI = micI * scale;
            
            cleanMix = micT + micI;
            
            % Add White Noise (SNR = 5dB)
            powMix = mean(cleanMix(:).^2);
            noise = sqrt(powMix / (10^(snr_dB/10))) * randn(size(cleanMix));
            finalMix = cleanMix + noise;
            
            % --- D. Saving ---
            fName = sprintf('sample_%05d', i);
            subFolder = fullfile(outputDatasetDir, fName);
            if ~exist(subFolder, 'dir'), mkdir(subFolder); end
            
            % Normalize to prevent clipping
            peak = max(abs(finalMix(:)));
            if peak > 0.99, finalMix = finalMix * (0.99/peak); end
            
            audiowrite(fullfile(subFolder, 'mixture.wav'), finalMix, fs);
            audiowrite(fullfile(subFolder, 'target.wav'), targetSignal, fs);
            
            % --- SAVING INTERFERENCE (ADDED) ---
            audiowrite(fullfile(subFolder, 'interference.wav'), micI, fs);
            
            % --- E. Metadata ---
            meta = struct(); 
            meta.target_angle = targetAngle_deg;
            meta.interf_angle = interfererAngle_deg;
            meta.rt60         = beta;
            meta.source_class = sourceType; % Added: Male
            meta.interf_class = interfType; % Added: Female/Music/Noise
            meta.note         = "Fixed Geometry";
            
            fid = fopen(fullfile(subFolder, 'meta.json'), 'w');
            fprintf(fid, '%s', jsonencode(meta, 'PrettyPrint', true));
            fclose(fid);
            
            % Update progress
            send(q, i);
            
        catch ME
            fprintf('Error in sample %d: %s\n', i, ME.message);
        end
    end
    fprintf('Dataset Generation Complete.\n');
end




%% --- HELPER FUNCTIONS ---


function updateProgress(~, total)
    persistent pCount
    if isempty(pCount), pCount = 0; end
    pCount = pCount + 1;
    if mod(pCount, 100) == 0
        fprintf('Progress: %d / %d\n', pCount, total);
    end
end


function files = getFiles(root)
    if strlength(root) < 1, files = {}; return; end
    f = dir(fullfile(root, '**', '*.wav'));
    f2 = dir(fullfile(root, '**', '*.flac'));
    f = [f; f2];
    files = fullfile({f.folder}, {f.name});
end


function pos = getPositionFromAngle(angle_deg, r, c)
    rad = deg2rad(angle_deg);
    pos = [c(1)+r*cos(rad), c(2)+r*sin(rad), c(3)];
end


function out = preprocessAudio(p, fs, samps)
    [x, fIn] = audioread(p);
    % Resample if needed
    if fIn ~= fs, x = resample(x, fs, fIn); end
    % Convert to mono
    if size(x, 2) > 1, x = x(:, 1); end
    
    % Pad or Crop
    if length(x) < samps
        out = [x; zeros(samps - length(x), 1)];
    else
        st = randi(length(x) - samps + 1);
        out = x(st : st + samps - 1);
    end
end
