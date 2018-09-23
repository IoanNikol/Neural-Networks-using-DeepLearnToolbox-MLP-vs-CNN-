% =========================================================================
% =============== ARTIFICIAL INTELLIGENCE : CNN vs MLP ====================
% =================== CNN : i - 6c - 2s - 12s - 2s ========================
% =========================================================================
clc;
clear all;

path = '...'; % images path
cd(path);

% ===================== PRE-PROSCESSING PHASE =============================

I = eye(40);

tr_faces_x = zeros(32, 32, 320); % image size : 32 x 32
tst_faces_x = zeros(32, 32, 80);

tr_faces_y = zeros(40, 320);
tst_faces_y = zeros(40, 80);

tr_index = 1;
tst_index = 1;

for i = 1 : 40
    cd(path);
    dir_name = strcat(['s', num2str(i)]);
    cd(dir_name);
    
    for j = 1 : 10
        
        fname = strcat([num2str(j), '.pgm']);
        im = imread(fname);
        imcr = im(11:end-10, :);
        im32 = imresize(im, [32, 32], 'bicubic');
        im32 = double(im32);
        im32_col = im32(:);
        max = im32_col(1);
        min = im32_col(1);
        
        for k = 1 : length(im32_col) 
            if im32_col(k) > max
            max = im32_col(k);
            end    
        end 
        
        for l = 1 : length(im32_col)
            if im32_col(l) < min
                min = im32_col(l);
            end
        end
        im_norm = 2 * (im32 - min) / (max - min) - 1;
        
        if (j < 9)
            
            tr_faces_x(:, :, tr_index) = im_norm;
            tr_faces_y(:, tr_index) = I(:, i);
            tr_index = tr_index + 1;
            
        else 
            
            tst_faces_x(:, :, tst_index) = im_norm;
            tst_faces_y(:, tst_index) = I(:, i);
            tst_index = tst_index + 1;
        end    
        
    end    
end

% ====================== END OF PRE-PROCESSING PHASE=======================


path2 = '...'; % DeepLearningToolbox path
cd(path2);
rand('state',0)

% CNN Architecture
cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 24, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
    };

batch_size_options = [40 80 320]; % Define the desired batch size options
epoch_options = [100 200 300 400 500]; % Define the desired epoch for training
my_test_error = [];
my_train_error = [];

for i = 1 : length(batch_size_options)
    
    for j = 1 : length(epoch_options)
        
        opts.alpha = 1;
        opts.batchsize = batch_size_options(i);
        opts.numepochs = epoch_options(j);
        cnn = cnnsetup(cnn, tr_faces_x, tr_faces_y);
        cnn = cnntrain(cnn, tr_faces_x, tr_faces_y, opts);

        [er1, bad1] = cnntest(cnn, tst_faces_x, tst_faces_y);
        my_test_error(end + 1) = er1;

        [er2, bad2] = cnntest(cnn, tr_faces_x, tr_faces_y);
        my_train_error(end + 1) = er2;
    end
end


% Plot the errors
figure;
epochs = [100 200 300 400 500];
my_test_err1 = [my_test_error(1), my_test_error(2), my_test_error(3), my_test_error(4), my_test_error(5)];
my_train_err1 = [my_train_error(1), my_train_error(2), my_train_error(3), my_train_error(4), my_train_error(5)];

my_test_err2 = [my_test_error(6), my_test_error(7), my_test_error(8), my_test_error(9), my_test_error(10)];
my_train_err2 = [my_train_error(6), my_train_error(7), my_train_error(8), my_train_error(9), my_train_error(10)];

my_test_err3 = [my_test_error(11), my_test_error(12), my_test_error(13), my_test_error(14), my_test_error(15)];
my_train_err3 = [my_train_error(11), my_train_error(12), my_train_error(13), my_train_error(14), my_train_error(15)];

subplot(1, 3, 1);
title('batch size : 40');
plot(epochs, my_train_err1, epochs, my_test_err1);
legend('train error', 'test error');
ylabel('ERROR');
xlabel('epochs');

subplot(1, 3, 2);
title('batch size : 80');
plot(epochs, my_train_err2, epochs, my_test_err2);
legend('train error', 'test error');
ylabel('ERROR');
xlabel('epochs');

subplot(1, 3, 3);
title('batch size : 320');
plot(epochs, my_train_err3, epochs, my_test_err3);
legend('train error', 'test error');
ylabel('ERROR');
xlabel('epochs');