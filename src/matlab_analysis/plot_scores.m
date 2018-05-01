function  plot_scores(input,title)
disp(title)
    for i = 6:11
        plot(input(:,1), input(:,i))
        hold on
    end
    hold off
    axis([0 1 0 1])
    legend('MCC','F1-Score', 'F2-Score', 'Acc', 'Sens', 'Spec')
    ylabel('Score')
    xlabel('Threshold')
end 

