
x = [1 2 3];
 col = ['r','b']
vals = [c(1,1) c(1,2); c(2,1) c(2,2) ;c(3,1) c(3,2)];
b = bar(x,vals,  'FaceColor','b');
b(1).FaceColor = [0 0.4470 0.7410];
b(2).FaceColor = [0.6350 0.0780 0.1840];
% set(gca, 'YLim',f);
disp(length(c))
% for i = 1:length(c)
%     bar(c(i,:),'DisplayName','r')
% end
set(gca, 'xticklabel',{'SDR','SIR','SAR'},'fontsize', 12);
h=legend( 'Direct','Mask');
set(h,'Fontsize',16);
% set(gca, 'YLim',f);
FontSize = 20
set(gca,'FontWeight','bold','fontsize',FontSize,'Fontname','Times New Roman');
set(gca, 'YLim',[0,20]);