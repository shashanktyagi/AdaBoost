function [indx1,indx2, twin] = argmax(bu,w)
    w = reshape(w,[length(w),1,1]);
    s = squeeze(sum(bu.*w,1));
    [~,i] = max(abs(s(:)));
    twin = 1;
    [indx1,indx2] = ind2sub(size(s),i);
    if s(indx1,indx2) < 0
        twin = -1;
    end
end