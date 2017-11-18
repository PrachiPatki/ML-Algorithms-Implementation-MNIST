clear
load MNIST_digit_data
whos
I = 0;
rand('seed', 1);
inds = randperm(size(images_train, 1));
images_train = images_train(inds, :);
labels_train = labels_train(inds, :);

inds = randperm(size(images_test, 1));
images_test = images_test(1:1000, :);
labels_test = labels_test(1:1000, :); 

% 
I = find(labels_train==1);
x1 =  images_train(I,:);
y1 = labels_train(I,:);
I = find(labels_train==6);
x2 =  images_train(I,:);
y2 = labels_train(I,:);
y = [y1(1:500,:); y2(1:500,:)];
y(y==1)=1;
y(y==6)=-1;
x = [x1(1:500,:); x2(1:500,:)];

rand('seed', 1);
inds = randperm(size(x, 1));
x = x(inds, :);
y = y(inds, :);

images_train = x(1:1000, :);
labels_train = y(1:1000, :);

images_test = [x1(501:1000,:); x2(501:1000,:)];
labels_test = [y1(501:1000,:); y2(501:1000,:)];
labels_test(labels_test==1)=1;
labels_test(labels_test==6)=-1;

images_test = images_test(inds, :);
labels_test = labels_test(inds, :);

[row,column] = size(images_train);
[row_test,column_test] = size(images_test);
pos = zeros(1,column);
neg = zeros(1,column);

weights = zeros(1,column);
bias=0;
iterations = 1;
[weights, bias, accuracy] = Perceptron(iterations, images_train,labels_train,images_test,labels_test,weights, bias);
%[accuracy] = Accuracy(output,labels_test);
figure;
plot([1:1000*iterations], accuracy);
xlabel('No. of Iterations')
ylabel('Accuracy')
title('Q5_C')


[pos_reshape,neg_reshape,col,row] = plotting(weights, pos ,neg);
 figure;
 imshow(pos_reshape)
 title('Reshaped image 1')
 figure;
 imshow(neg_reshape)
 title('Reshaped image 6')
 %figure;
 %scatter(pos_reshape); 

function [weights, bias, accuracy] = Perceptron(iterations, images_train,labels_train,images_test,labels_test,weights, bias)
[row, column] = size(images_train);
output = zeros(size(labels_test));
accuracy = zeros(1,row*iterations);
    for i = 1:iterations
        for j = 1:row
           activation = dot(weights, images_train(j,:))+bias;
            if labels_train(j)*activation <= 0
                temp =  labels_train(j)*images_train(j,:);
                weights = temp + weights;
                bias= bias + labels_train(j);
            end
            output = Perceptron_test(weights,bias,images_test, labels_test);
            accuracy((i-1)*row+j) = Accuracy(output,labels_test);
        end
    end    

end


function [pos_reshape, neg_reshape,col,row] = plotting(weights, pos ,neg)
pos_reshape = zeros(28,28);
neg_reshape = zeros(28,28);
    for j = 1:784
        if weights(j)<0
            neg(j) = 255;
            neg_reshape = reshape(neg,[28,28]);
        elseif weights(j)>0
            pos(j) = weights(j);
            pos_reshape = reshape(pos,[28,28]); 
            [row, col] = find(pos_reshape);
             
        end
    end

end




function [output] = Perceptron_test(weights,bias,images_test, labels_test)
  [row_test,column_test] = size(images_test);  
        output = zeros(size(labels_test));
        for j = 1:row_test
           activation = dot(weights, images_test(j,:))+bias;
            if activation < 0.0
                output(j) = -1;
            elseif activation > 0.0
                output(j) = 1;
            else 
                display(activation);
            end
            
        end
    
end



function [accuracy] = Accuracy(output,labels_test)
     a = output == labels_test;
     x = (sum(a(:)==1));
     accuracy = x * 100/ 1000;
end