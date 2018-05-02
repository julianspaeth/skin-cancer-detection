function  plot_scores(input,title)
    for i = 6:11
        plot(input(:,1), input(:,i))
        hold on
    end
    hold off
    axis([0 1 0 1])
    leg = legend('MCC','F1-Score', 'F2-Score', 'Acc', 'Sens', 'Spec',  'Location', 'SouthEast');
    leg.FontSize=18;
    ylabel('Score', 'FontSize',30)
    xlabel('Threshold', 'FontSize',30)
end 

