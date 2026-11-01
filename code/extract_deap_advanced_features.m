% extract_deap_advanced_features.m
% Extracts Differential Entropy (DE) features from DEAP dataset without requiring Signal Processing Toolbox
% Features extracted per 2s window for 14 channels across 4 frequency bands (Theta, Alpha, Beta, Gamma)

clear; clc;
disp('Starting DEAP Advanced Feature Extraction (DE)...');

% Config
data_dir = 'Z:\brain\brain-data\DEAP\data_preprocessed_matlab\';
% Generate subject filenames dynamically
subjects = cell(1, 32);
for i = 1:32
    subjects{i} = sprintf('s%02d.mat', i);
end 
channels = [2, 3, 4, 5, 7, 12, 14, 18, 20, 21, 22, 26, 30, 32];
fs = 128; % Sampling frequency
win_len = 256; % 2 seconds

% Frequency bands
bands = [4, 8;   % Theta
         8, 13;  % Alpha
         13, 30; % Beta
         30, 45];% Gamma

all_features = [];
all_labels = [];

for s = 1:length(subjects)
    disp(['Processing ', subjects{s}]);
    load(fullfile(data_dir, subjects{s}));
    
    for trial = 1:40
        trial_data = squeeze(data(trial, channels, :)); % 14 x 8064
        trial_label = squeeze(labels(trial, :)); % 1 x 4
        
        num_windows = floor(8064 / win_len);
        for w = 1:num_windows
            start_idx = (w-1)*win_len + 1;
            end_idx = w*win_len;
            window_data = trial_data(:, start_idx:end_idx); % 14 x 256
            
            % Compute FFT
            N = win_len;
            freqs = (0:N-1)*(fs/N);
            X = fft(window_data, [], 2);
            P = (abs(X).^2) / N; % Power spectrum
            
            % Compute DE for each band
            % DE = 0.5 * log(2 * pi * e * variance) -> variance ~ band power
            de_features = zeros(length(channels), size(bands, 1));
            for b = 1:size(bands, 1)
                % Find indices for the band
                idx = freqs >= bands(b, 1) & freqs <= bands(b, 2);
                band_power = sum(P(:, idx), 2);
                
                % Add small epsilon to avoid log(0)
                band_power(band_power == 0) = 1e-10;
                
                % Calculate DE
                de_features(:, b) = 0.5 * log(2 * pi * exp(1) * band_power);
            end
            
            % Flatten DE features: 14 channels x 4 bands = 56 features
            all_features = [all_features; de_features(:)'];
            all_labels = [all_labels; trial_label];
        end
    end
end

disp(['Total samples extracted: ', num2str(size(all_features, 1))]);

% Save to MAT file to be loaded by Python later
save('de_features_all.mat', 'all_features', 'all_labels', '-v7.3');
disp('Features saved to de_features_all.mat successfully!');
