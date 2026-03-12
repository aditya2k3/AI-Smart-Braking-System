%% Smart Braking System - GPU Option Corrected
clc;
clear;
rng(42); % For reproducibility

% ========================
% SYSTEM CONFIGURATION
% ========================
enableGPU = true; % Boolean flag for GPU usage
useGPUOption = 'no'; % Default to CPU
maxNumCompThreads(16); % Utilize 16 CPU threads

% Check GPU availability
if enableGPU && (gpuDeviceCount > 0)
    try
        g = gpuDevice;
        fprintf('Using %s GPU with %.1f GB VRAM\n', g.Name, g.AvailableMemory/1e9);
        useGPUOption = 'yes'; % Set to use GPU
    catch
        warning('GPU access failed - falling back to CPU');
        enableGPU = false;
    end
end

% ========================
% ENHANCED DATA GENERATION
% ========================
[num_samples, time, vehicle_speed, obstacle_distance, ...
    road_friction, camera_detection] = generate_enhanced_sensor_data();

% Physics-based ground truth with camera input
deceleration = 0.8 * road_friction * 9.81;
true_braking_distance = (vehicle_speed.^2) ./ (2 * deceleration);
true_braking_distance(camera_detection > 0.7) = true_braking_distance(camera_detection > 0.7) .* 0.85;

% Sensor fusion with explicit array sizing
sensor_distance = 0.3*obstacle_distance.*(1 + 0.02*randn(size(obstacle_distance))) + ...
                  0.2*obstacle_distance.*(1 + 0.03*randn(size(obstacle_distance))) + ...
                  0.5*obstacle_distance.*(1 + 0.01*randn(size(obstacle_distance)));

% ========================
% GPU-OPTIMIZED ANN TRAINING
% ========================
input_data = [vehicle_speed; sensor_distance; road_friction; camera_detection]';
output_data = true_braking_distance';

% Configure ANN with automatic GPU handling
net = fitnet([128 64 32], 'trainlm');
net.trainParam.epochs = 1000;
net.trainParam.max_fail = 15;
net.performParam.regularization = 0.001;
net.divideParam.trainRatio = 0.75;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.10;

% Train network with proper GPU option
[net, tr] = train(net, input_data', output_data', ...
    'useGPU', useGPUOption, ...  % Now using correct string value
    'showResources', 'yes', ...
    'reduction', 1);

% ANN prediction
ann_predicted_distance = net(input_data')';

% ========================
% REMAINING CODE (PID and visualization as before)
% ========================
[pid_braking_force, pid_stats] = adaptive_pid_controller(...
    sensor_distance, ann_predicted_distance, time, enableGPU);

create_interactive_dashboard(time, true_braking_distance, ...
    ann_predicted_distance, sensor_distance, pid_braking_force, ...
    vehicle_speed, road_friction, camera_detection);

% ========================
% CUSTOM FUNCTIONS (Remain unchanged from previous version)
% ========================
% ========================
% CUSTOM FUNCTIONS
% ========================
function [num_samples, time, speed, distance, friction, camera] = generate_enhanced_sensor_data()
    num_samples = 1e5; % Leverage system RAM capacity
    time = linspace(0, 20, num_samples);
    
    % Enhanced vehicle dynamics model
    speed = 20 + 5*sin(linspace(0, 4*pi, num_samples)) + ...
        2*randn(1, num_samples);
    
    % Obstacle distance with multiple emergency scenarios
    distance = 100 - cumsum(0.2 + 0.1*randn(1, num_samples));
    emergency_points = randperm(num_samples, 50);
    distance(emergency_points) = distance(emergency_points) - 25;
    
    % Road friction with sudden changes
    friction = 0.4 + 0.4*tanh(linspace(-5, 5, num_samples)) + ...
        0.05*randn(1, num_samples);
    friction(50000:50050) = 0.15; % Simulate ice patch
    
    % Camera detection (synthetic image processing output)
    camera = 0.7*sigmoid(linspace(-5, 5, num_samples)) + ...
        0.3*randn(1, num_samples);
    camera = max(min(camera, 1), 0);
    
    % Nested sigmoid function
    function y = sigmoid(x)
        y = 1./(1 + exp(-x));
    end
end

function [pid_force, stats] = adaptive_pid_controller(sensor, ann, t, useGPU)
    % GPU-aware PID implementation
    if useGPU
        sensor = gpuArray(sensor);
        ann = gpuArray(ann);
        t = gpuArray(t);
    end
    
    Kp = 0.9; Ki = 0.05; Kd = 0.3;
    pid_force = zeros(size(sensor), 'like', sensor);
    integral = 0;
    prev_error = 0;
    
    error_history = zeros(1, 3);
    
    for i = 1:length(sensor)
        error = sensor(i) - ann(i);
        error_history = [error_history(2:end) error];
        
        % Adaptive tuning based on error dynamics
        if std(error_history) > 0.1
            Kp_adjust = 1.2;
            Ki_adjust = 0.8;
        else
            Kp_adjust = 1.0;
            Ki_adjust = 1.0;
        end
        
        integral = integral + error;
        derivative = (error - prev_error) / (t(2)-t(1));
        
        % Conditional integration anti-windup
        if abs(integral) > 50
            integral = sign(integral) * 50;
        end
        
        pid_force(i) = (Kp*Kp_adjust)*error + ...
                      (Ki*Ki_adjust)*integral + ...
                      Kd*derivative;
        prev_error = error;
    end
    
    stats.max_force = max(pid_force);
    stats.avg_force = mean(pid_force);
end

function create_interactive_dashboard(t, true_dist, ann_dist, ...
    sensor_dist, pid_force, speed, friction, camera)
    
    fig = figure('Position', [100 100 1440 900], 'Color', 'w');
    
    % 3D Surface Plot: Speed-Friction-Distance
    subplot(3,3,[1 2 4 5]);
    [X,Y] = meshgrid(linspace(min(speed),max(speed),50), ...
                   linspace(min(friction),max(friction),50));
    Z = griddata(speed, friction, true_dist, X, Y);
    surf(X,Y,Z, 'EdgeColor', 'none');
    hold on;
    scatter3(speed, friction, ann_dist, 10, 'filled', 'MarkerFaceColor', 'r');
    title('3D Braking Dynamics');
    xlabel('Speed (m/s)');
    ylabel('Friction');
    zlabel('Distance (m)');
    colormap jet;
    colorbar;
    view(-45,30);
    
    % ANN Performance Metrics
    subplot(3,3,3);
    residuals = true_dist - ann_dist;
    histogram(residuals, 50, 'FaceColor', [0.2 0.6 0.8]);
    title(sprintf('ANN Residuals\nμ=%.3f σ=%.3f', mean(residuals), std(residuals)));
    grid on;
    
    % Real-time Simulation View
    subplot(3,3,6);
    plot(t, speed, 'b', t, friction*20, 'm');
    title('Vehicle State');
    xlabel('Time (s)');
    ylabel('Speed (m/s) | Friction*20');
    legend('Speed', 'Friction (scaled)');
    ylim([0 35]);
    grid on;
    
    % Sensor Fusion Analysis
    subplot(3,3,[7 8 9]);
    yyaxis left;
    plot(t, true_dist, 'k--', t, ann_dist, 'r', t, sensor_dist, 'b:');
    ylabel('Distance (m)');
    yyaxis right;
    plot(t, pid_force, 'g', 'LineWidth', 1.5);
    ylabel('Braking Force (N)');
    title('Integrated System Performance');
    legend('True Distance', 'ANN Prediction', 'Sensor Reading', 'PID Force');
    grid on;
    
    % Camera Detection Overlay
    annotation(fig, 'rectangle', [0.78 0.82 0.18 0.12], 'Color', 'k');
    annotation(fig, 'textbox', [0.78 0.82 0.18 0.12], 'String', ...
        sprintf('Camera Detection Confidence\nMean: %.2f\nMax: %.2f', ...
        mean(camera), max(camera)), 'EdgeColor', 'none');
end
