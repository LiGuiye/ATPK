function [Z_ATPK]=ATPK_Guiye(lores,scale,Sill_min,Range_min,L_sill,L_range,rate,max_lag,W,PSF)
RB_extend1=[repmat(lores(:,1),[1,W]),lores,repmat(lores(:,end),[1,W])];%%%extend columns
RB_extend=[repmat(RB_extend1(1,:),[W,1]);RB_extend1;repmat(RB_extend1(end,:),[W,1])];%%%extend rows

x0=[0.1,1];%%%%%x0 is the initial value for fitting
for h=1:max_lag
    rh(h)=semivariogram(lores,h);
end
[xa1,resnorm]=lsqcurvefit(@myfun2,x0,scale:scale:scale*max_lag,rh);
xp_best=ATP_deconvolution0(max_lag,scale,xa1,Sill_min,Range_min,L_sill,L_range,rate);

yita1=ATPK_noinform_yita_new(scale,W,xp_best,PSF);
P_vm=ATPK_noinform_new(scale,W,RB_extend,yita1);
Z_ATPK=P_vm(W*scale+1:end-W*scale,W*scale+1:end-W*scale);
