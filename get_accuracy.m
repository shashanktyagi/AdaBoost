function error = get_accuracy(X_tr,weak_learner,w,binary_labels,T)
g=0;
for t = 1:T
    pred(X_tr(:,weak_learner(t,2)) < weak_learner(t,1)) = -1;
    pred = weak_learner(t,3)*pred;
    g = g + w(t)*pred; 
end
error = sum(sign(g)~=binary_labels)/N;