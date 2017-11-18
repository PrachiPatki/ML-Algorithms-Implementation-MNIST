clear
load MNIST_digit_data
whos
I = 0;
rand('seed', 1);
inds = randperm(size(images_train, 1));
images_train = images_train(inds, :);
labels_train = labels_train(inds, :);

% for classification 1 & 6

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

% Final train and test set

images_train = x(1:1000, :);
labels_train = y(1:1000, :);

images_test = [x1(501:1000,:); x2(501:1000,:)];
labels_test = [y1(501:1000,:); y2(501:1000,:)];
labels_test(labels_test==1)=1;
labels_test(labels_test==6)=-1;

images_test = images_test(inds, :);
labels_test = labels_test(inds, :);


[row,column] = size(images_train);
weights = zeros(1,column);
iterations = 1;
n = 1/iterations;

[weights,accuracy] = SupportVector(iterations, images_train,labels_train,images_test,labels_test,weights);
figure;
plot(1:1000, accuracy);

function[weights,accuracy] = SupportVector(iterations, images_train,labels_train,images_test,labels_test,weights)
[row, column] = size(images_train);
output = zeros(size(labels_test));
accuracy = zeros(1,row);
C = 0.01;
count=0;


    for i = 1:iterations
        
        for j = 1:row
            count= count+1;
            n = 1/count;
            y_updated = dot(weights,images_train(j,:));
            if (labels_train(j)* y_updated) > 1
                 weights = (1 - n * C ) * weights;
            elseif(labels_train(j)* y_updated) <= 1           
                temp = n * labels_train(j) * images_train(j,:);
                weights = temp + weights;
            end
            output = SVM_test(weights,images_test, labels_test);
            accuracy(j) = Accuracy(output,labels_test);
        end
        
    end
end



function [output]= SVM_test(weights,images_test,labels_test)
    [row_test,column_test] = size(images_test);
      output = zeros(size(labels_test));
      for j = 1:row_test
        y_updated = dot(weights,images_test(j,:));
        output(j) = y_updated;
      end
end
    
function [accuracy] = Accuracy(output,labels_test)
      a = zeros(size(output));
      correct=0;
      [row_test,column_test] = size(labels_test);
      for j = 1:row_test
           % a(j)=((labels_test(j)*output(j))>=1);
           if (labels_test(j)*output(j))>1
                correct= correct+1;
           end
      end
      accuracy = (correct*100)/row_test;
end




    
    
    
    
    
    
    
    
    
    
    
   
    

