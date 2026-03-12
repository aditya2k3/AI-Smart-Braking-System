% --- Bird's-Eye View Plot (Adjusted for Compatibility) ---
if license('test', 'Automated_Driving_Toolbox')
    try
        % Extract velocity (m/s) and obstacle distance (m) from X_val
        % Ensure columns 1 and 2 are velocity and distance (adjust if needed!)
        velocity = X_val(:,1);       % Column 1: Velocity
        distance_obstacle = X_val(:,2); % Column 2: Distance to obstacle
        
        % Simulate time steps (1 second per sample)
        time = 0:length(velocity)-1;
        egoX = cumsum(velocity .* 1); % Simplified position over time
        
        % Initialize bird's-eye plot
        figure;
        bep = birdsEyePlot('XLim', [0 max(egoX)+50], 'YLim', [-10 10]);
        title(bep.Parent, 'Bird''s-Eye View with Braking Force');
        
        % Create ego vehicle and obstacle
        egoCar = actor(bep, 'Position', [egoX(1) 0 0], 'Length', 4.7, 'Width', 1.8, 'Color', [0 0.7 0.7]);
        obstacle = actor(bep, 'Position', [egoX(1)+distance_obstacle(1) 0 0], 'Length', 4.7, 'Width', 1.8, 'Color', [0.9 0 0]);
        
        % Plot key frames (first 20 samples)
        numFrames = min(20, length(time));
        for i = 1:numFrames
            % Update positions
            egoCar.Position = [egoX(i), 0, 0];
            obstacle.Position = [egoX(i) + distance_obstacle(i), 0, 0];
            
            % Display braking force
            annotationText = sprintf('Braking Force:\nActual = %.1f\nPredicted = %.1f', Y_val(i), Y_pred(i));
            text(bep.Parent, 0.5, 0.95, annotationText, 'Units', 'normalized', 'Color', 'red', 'FontSize', 10);
            
            % Pause to visualize
            pause(0.1);
        end
    catch ME
        warning('Bird''s-eye view failed: %s', ME.message);
    end
else
    warning('Automated Driving Toolbox not available. Skipping bird''s-eye view.');
end