close all
[X_tr, Y_tr] = readMNIST('./data/train-images-idx3-ubyte', './data/train-labels-idx1-ubyte',20000,0);
[X_te, Y_te] = readMNIST('./data/t10k-images-idx3-ubyte', './data/t10k-labels-idx1-ubyte',10000,0);
fprintf('Data loaded...\n')

%%
num_classes = 10;
[N,D] = size(X_tr);
T = 250;
parameters = ones(51,D).*(0:50)'/50;
weak_learner = zeros(T,3,num_classes);
weights = zeros(N,T,num_classes);
w = zeros(T,num_classes);
e = zeros(T,1);
margin = zeros(N,T,num_classes);
l_w_indx = zeros(T,num_classes);
u = decision_stump(X_tr,parameters);
for i = 1:num_classes
    fprintf('Learning classifier for digit %d ...\n',i-1)
    binary_labels = -ones(N,1);
    binary_labels(Y_tr == i-1) = 1;
    bu = reshape(binary_labels,[length(binary_labels),1,1]).*u;
    g = zeros(N,1);
    fprintf('Starting iterations...\n') 
    for t = 1:T
       margin(:,t,i) = binary_labels.*g;
       weights(:,t,i) = exp(-margin(:,t,i));
       [~,l_w_indx(t,i)] = max(weights(:,t,i));
       [temp, weak_learner(t,2,i), weak_learner(t,3,i)] = argmax(bu,weights(:,t,i));
       weak_learner(t,1,i) = parameters(temp,1);
       pred = ones(N,1);
       pred(X_tr(:,weak_learner(t,2,i)) < weak_learner(t,1,i)) = -1;
       pred = weak_learner(t,3,i)*pred;
       e(t) = sum(weights(pred ~= binary_labels,t,i))/sum(weights(:,t,i));
       w(t,i) = 0.5*log((1-e(t))/e(t));
       g = g + w(t,i)*pred;
       if rem(t,50) == 0
           fprintf('t: %d/%d  e: %f\n',t,T,e(t))
       end
    end  
end
save

%%
error = zeros(T,num_classes);
error2 = zeros(T,num_classes);
[N,~] = size(X_tr);
[N2,~] = size(X_te);
g = zeros(N,num_classes);
g2 = zeros(N2,num_classes);
for i=1:num_classes
    binary_labels = -ones(N,1);
    binary_labels(Y_tr == i-1) = 1;
    for t = 1:T        
        pred = ones(N,1);
        pred(X_tr(:,weak_learner(t,2,i)) < weak_learner(t,1,i)) = -1;
        pred = weak_learner(t,3,i)*pred;
        g(:,i) = g(:,i) + w(t,i)*pred;   
        error(t,i) = sum(sign(g(:,i))~=binary_labels)/N;
    end
    
    binary_labels2 = -ones(N2,1);
    binary_labels2(Y_te == i-1) = 1;
    for t2 = 1:T        
        pred2 = ones(N2,1);
        pred2(X_te(:,weak_learner(t2,2,i)) < weak_learner(t2,1,i)) = -1;
        pred2 = weak_learner(t2,3,i)*pred2;
        g2(:,i) = g2(:,i) + w(t2,i)*pred2;   
        error2(t2,i) = sum(sign(g2(:,i))~=binary_labels2)/N2;
    end
    
    figure
    plot(error(:,i));
    hold on
    plot(error2(:,i));
    xlabel('iterations');
    ylabel('error');
    legend('Training error','Test error')
    saveas(gcf,sprintf('tr%d.eps',i-1),'epsc')
    close all
    
end 

% [~,pred] = max(g,[],2);
% total_error = 1-sum(Y_tr == (pred-1))/N;
% 
% plot(error)
% xlabel('iterations')
% ylabel('error')
%%

t = [5,10,50,100,250];
m = margin(:,t,:);
nbins = 10;
h = zeros(nbins,length(t));
cdf = zeros(nbins,length(t));
% edges = zeros(nbins,length(t));
for k=1:10
    figure
for i=1:length(t)
%     [h(:,i), edges] = histcounts(m(:,i,1),nbins);
%     cdf(:,i) = cumsum(h(:,i)/sum(h(:,i)));
    cdfplot(m(:,i,k));
    hold on
end
legend('iteration = 5','iteration = 10', 'iteration = 50', 'iteration = 100', 'iteration = 250')
saveas(gcf,sprintf('cdf%d.eps',k-1),'epsc')
close all
end
%plot(cdf)
legend('iteration = 5','iteration = 10', 'iteration = 50', 'iteration = 100', 'iteration = 250')
%%
for i =1:10
plot(l_w_indx(:,i))
xlabel('iteration')
ylabel('index')
saveas(gcf,sprintf('lw%d.eps',i-1),'epsc')
close all
end
%%
heavy_ex = zeros(3,10);
for i=1:10
uq = unique(l_w_indx(:,i));
c = [uq,histc(l_w_indx(:,i),uq)];
[~,indx] = sort(c(:,2),'descend');
heavy_ex(:,i) = c(indx(1:3),1);
end

figure;
images = zeros(28*10,28*3);
for i = 1:10
    for j = 1:3   
        images((i-1)*28+1:(i-1)*28+28,(j-1)*28+1:(j-1)*28+28) = reshape(X_tr(heavy_ex(j,i),:),28,28)';
    end
end
imshow(images)


%%


for i = 1:2:10
    a = 128*ones(1,28*28);
    for t=1:T
        a(weak_learner(t,2,i)) = 255*(weak_learner(t,3,i)+1)*0.5;
    end
    subplot(5,2,i)
    imshow(reshape(a,28,28)'/255)
    xlabel(sprintf('Classifier for digit %d',i-1))
    a = 128*ones(1,28*28);
    for t=1:T
        a(weak_learner(t,2,i+1)) = 255*(weak_learner(t,3,i+1)+1)*0.5;
    end
    subplot(5,2,i+1)
    imshow(reshape(a,28,28)'/255)
    xlabel(sprintf('Classifier for digit %d',i))
end








