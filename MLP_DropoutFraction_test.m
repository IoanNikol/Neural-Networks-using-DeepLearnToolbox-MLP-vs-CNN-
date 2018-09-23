% =========================================================================
% ============== ARTIFICIAL INTELLIGENCE : CNN vs MLP =====================
% ================ MLP : dropout fraction selection =======================
% =========================================================================
clc;
clear all;

path = ''; % images path
cd(path);

% ===================== PRE-PROSCESSING PHASE =============================

I = eye(40);

tr_faces_x = zeros(32, 32, 320);
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
        im = im2double(im);
        imcr = im(11:end-10, :);
        im32 = imresize(im, [32, 32], 'bicubic');
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

% Store the images into columns 
[r, c, n] = size(tr_faces_x);
tr_faces_x = reshape(tr_faces_x, r*c, n)';
tr_faces_y = tr_faces_y';

[r, c, n] = size(tst_faces_x);
tst_faces_x = reshape(tst_faces_x, r*c, n)';
tst_faces_y = tst_faces_y';

% ====================== END OF PRE-PROCESSING PHASE=======================
path2 = '...'; % DeepLearnToolbox
cd(path2);

rand('state',0)

dropout_options = [0 0.5]; % Define dropout options
count = length(dropout_options);

my_train_error = [];
my_test_error = [];

for i = 1 : count
    
    nn = nnsetup([r*c 400 40]);
    
    nn.activation_function = 'sigm';    %  Choose : 'sigm', 'tanh' or 'tahn_opt'
    nn.output = 'sigm';              % Choose : 'sigm' or 'softmax'
    nn.momentum = 0.1;
    nn.dropoutfraction = dropout_options(i);
    nn.learningRate = 0.1;              %  Sigm require a lower learning rate
    opts.numepochs =  300;                %  Number of full sweeps through data
    opts.batchsize = 80;                %  Take a mean gradient step over this many samples
    
    nn = nntrain(nn, tr_faces_x, tr_faces_y, opts);

    % Find the training and testing error
    [er1, bad1] = nntest(nn, tr_faces_x, tr_faces_y);
    [er2, bad2] = nntest(nn, tst_faces_x, tst_faces_y);
    my_train_error = [my_train_error, er1];
    my_test_error = [my_test_error, er2];
end

figure;
plot(dropout_options, my_train_error, dropout_options, my_test_error);
legend('train error', 'test error');
ylabel('ERROR');
xlabel('dropout fraction')