clc; close all;

%% 1. CONFIGURATION
maleDir   = "../Dataset_raw/Male";           
femaleDir = "../Dataset_raw/Female";         
noiseDir  = "../Dataset_raw/Noise";    
musicDir  = "../Dataset_raw/Music";   

% Output folder
outputDatasetDir = '../Test_Dataset/reverb';

% --- Generation Settings ---
numSamplesToGenerate = 150000;
desiredLength_s      = 4.0;
fs                   = 16000;
desiredLength_samples = floor(desiredLength_s * fs);

% --- Acoustic Settings ---
sir_dB            = 0;     % Signal-to-Interference Ratio
snr_dB            = 5;     % Signal-to-Noise Ratio
source_radius_m   = 1.0;
roomDimensions    = [4.9, 4.9, 4.9];

% --- FIXED ANGLES ---
targetAngle_deg     = 90;
interfererAngle_deg = 40;

% Mic Array (2 Mics)
micPositions      = [2.41, 2.45, 1.5;   
                     2.49, 2.45, 1.5];
micCenter         = mean(micPositions, 1);

c     = 343;   % Speed of sound
beta  = 0.5;   % RT60 
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
% NOTE: If these are all zeros (except index 1), you must run:
% mex rir_generator.cpp rir_generator_core.cpp
rirT_raw = rir_generator(c, fs, micPositions, posT, roomDimensions, beta, n_rir);
rirI_raw = rir_generator(c, fs, micPositions, posI, roomDimensions, beta, n_rir);

% --- DEBUG PLOT (ADDED) ---
% This plots the RIR so you can verify it is not blank
figure;
subplot(2,1,1); plot(rirT_raw(1,:)); title('Target RIR Channel 1 (Linear)'); grid on;
subplot(2,1,2); plot(10*log10(abs(rirT_raw(1,:)).^2 + 1e-12)); title('Target RIR Log Energy (dB)'); grid on;
drawnow;
fprintf('   - RIR Visualization plotted. Inspect figure window.\n');
% --------------------------

% Store in Constant memory
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
        sourceType = "Male"; 
        
        % Interferer Distribution
        roll = rand();
        interfType = ""; 
        
        if roll < (1/3)
            iList = cFemale.Value; 
            interfType = "Female";
        elseif roll < (2/3)
            iList = cMusic.Value;  
            interfType = "Music";
        else
            iList = cNoise.Value;  
            interfType = "Noise";
        end
        
        if isempty(iList), iList = cNoise.Value; interfType = "Noise"; end
        interfererFile = iList{randi(numel(iList))};
        
        % Process Audio
        targetSignal = preprocessAudio(targetFile, fs, desiredLength_samples);
        interfererSignal = preprocessAudio(interfererFile, fs, desiredLength_samples);
        
        % --- B. RIR Retrieval (Fixed) ---
        rirT = cRirTarget.Value;
        rirI = cRirInterf.Value;
        
        % Convolution
        micT = fftfilt(rirT', targetSignal);
        micI = fftfilt(rirI', interfererSignal);
        
        % Truncate
        len = min(size(micT, 1), size(micI, 1));
        micT = micT(1:len, :);
        micI = micI(1:len, :);
        
        % --- C. Mixing (SIR & SNR) ---
        powT = mean(micT(:,1).^2) + 1e-12;
        powI = mean(micI(:,1).^2) + 1e-12;
        
        % Adjust Interferer for SIR
        scale = sqrt(powT / powI) / (10^(sir_dB/20));
        micI = micI * scale;
        
        cleanMix = micT + micI;
        
        % Add White Noise for SNR
        powMix = mean(cleanMix(:).^2);
        noise = sqrt(powMix / (10^(snr_dB/10))) * randn(size(cleanMix));
        finalMix = cleanMix + noise;
        
        % --- D. Saving ---
        fName = sprintf('sample_%05d', i);
        subFolder = fullfile(outputDatasetDir, fName);
        if ~exist(subFolder, 'dir'), mkdir(subFolder); end
        
        % Normalize
        peak = max(abs(finalMix(:)));
        if peak > 0.99, finalMix = finalMix * (0.99/peak); end
        
        audiowrite(fullfile(subFolder, 'mixture.wav'), finalMix, fs);
        audiowrite(fullfile(subFolder, 'target.wav'), targetSignal, fs);
        audiowrite(fullfile(subFolder, 'interference.wav'), micI, fs);
        
        % --- E. Metadata ---
        meta = struct(); 
        meta.target_angle = targetAngle_deg;
        meta.interf_angle = interfererAngle_deg;
        meta.rt60         = beta;
        meta.source_class = sourceType;
        meta.interf_class = interfType;
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
    % Resample
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