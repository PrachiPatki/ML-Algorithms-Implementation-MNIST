clear
load MNIST_digit_data
whos



rand('seed', 1);
inds = randperm(size(images_train, 1));
images_train = images_train(inds, :);
labels_train = labels_train(inds, :);

%train_size = 1000; %%%%%vary this to change trainign size
inds = randperm(size(images_test, 1));
images_test = images_test(1:1000, :);
labels_test = labels_test(1:1000, :); 
k = 1;
a = size(labels_test);
index = 14;
A(images_train, labels_train, images_test(index,:), k, 60000); %A
%B(images_train, labels_train, images_test, labels_test, k); %B
%AvgAcc = C(images_train, labels_train, images_test, labels_test, k, 1); %c
%D(images_train, labels_train, images_test, labels_test, k) %D
%E(images_train(1:1000, :), labels_train(1:1000, :), images_train(1001:2000, :), labels_train(1001:2000, :)) %E

function E(images_train, labels_train, images_test, labels_test)
    
    for k = 1:10
        [acc, avv_Acc(k)] = kNN(images_train, labels_train, images_test, labels_test, k, 1000);
        ki(k) = k;
    end
    display(avv_Acc);
    figure;
    bar(ki, avv_Acc);
    xlabel('values of k')
    ylabel('Average Accuracy')
    
end

function D(images_train, labels_train, images_test, labels_test, k)
    ki = [1,2,3,5,10];
    for i  = 1:5
        acc_i = C(images_train, labels_train, images_test, labels_test, ki(i), 0);
        for j=1:10
           acc(i,j) = acc_i(j); 
        end
    end
    display(acc);
    train_size = logspace(log10(30),log10(10000),10);
    figure;
    plot(train_size, acc(1,:), train_size, acc(2,:), train_size, acc(3,:), train_size, acc(4,:), train_size, acc(5,:));
    legend('k(1)','k(2)','k(3)','k(5)','k(10)')
    xlabel('Training Data')
    ylabel('Accuracy')
end


function avv_Acc = C(images_train, labels_train, images_test, labels_test, k, disp)
    train_size = logspace(log10(30),log10(10000),10);
    avv_Acc = zeros(size(train_size));
    for i = 1:10
        [acc, avv_Acc(i)] = kNN(images_train, labels_train, images_test, labels_test, k, int16(train_size(i)));
    end
    %display(avv_Acc);
    if disp == 1
        figure;
        bar(train_size, avv_Acc);
        plot(train_size, avv_Acc);
        xlabel('Number of training data 30 to 10000')
        ylabel('Average Accuracy')
    end
end

function B(images_train, labels_train, images_test, labels_test, k)
    [Accuracy, Avg_Accuracy]= kNN(images_train, labels_train, images_test, labels_test, k, 60000);
    display(Accuracy);
    display(Avg_Accuracy);
end

function A(images_train, labels_train, image_test, k, train_size)
    [I, predict_num] = fitkNN(images_train, labels_train, image_test, k, train_size);
    show_img(predict_num, image_test);
end

function [acc, acc_av] = kNN(images_train, labels_train, images_test, labels_test, k, train_size)
    correct = zeros(1,10);
    total = zeros(1,10);
    acc = zeros(1,10);
    for i = 1:size(labels_test)
        point = images_test(i, :);
        [I, predict_val] = fitkNN(images_train, labels_train, point, k, train_size);
        if predict_val == labels_test(i)
            index = labels_test(i)+1;
            correct(index) = correct(index)+1;
        end
        total(labels_test(i)+1) = total(labels_test(i)+1)+1;
    end
   	for i = 1:10
        acc(i) = 100*correct(i)/total(i);
    end
    acc_av =  mean(acc);
end

function [I, predict_val] = fitkNN(images_train, labels_train, image_test, k, train_size)
    dist = pdist2(image_test, images_train(1:train_size,:));
    max_bin = zeros(1,10);
    for i = 1:k
        [M,I] = min(dist);
        [a, b] = max(dist);
        dist(I) = a;
        max_bin(labels_train(I)+1) = max_bin(labels_train(I)+1)+1;
    end
    [a, predict_val] = max(max_bin);
    predict_val = predict_val-1;
end

function show_img(predict_num, image)
    close all
    im = reshape(image(:), [28 28]);
    imshow(im)
    title(num2str(predict_num));
   
end