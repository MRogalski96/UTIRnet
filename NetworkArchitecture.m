function [lgraph] = NetworkArchitecture(num_filters,num_paths,imS,NoIn,NoOut)
%create layers
lgraph = layerGraph();
tempLayers = imageInputLayer([imS imS NoIn],"Name","imageinput");
lgraph = addLayers(lgraph,tempLayers);

for ii=1:num_paths
    if ii==1
        tempLayers = [
            convolution2dLayer([3 3],num_filters,"Name","conv1_path1","Padding","same")
            reluLayer("Name","relu_1_path1")];
    else
        tempLayers = [
            convolution2dLayer([3 3],num_filters,"Name","conv1_path"+num2str(ii),"Padding","same")
            reluLayer("Name","relu1_path"+num2str(ii))
            maxPooling2dLayer([2^(ii-1) 2^(ii-1)],"Name","maxpoolForUnpool1_path"+num2str(ii),"HasUnpoolingOutputs",true,"Padding","same","Stride",[2^(ii-1) 2^(ii-1)])];
    end
    lgraph = addLayers(lgraph,tempLayers);
    
    tempLayers = [
        convolution2dLayer([3 3],num_filters,"Name","conv2_path"+num2str(ii),"Padding","same")
        reluLayer("Name","relu2_path"+num2str(ii))
        convolution2dLayer([3 3],num_filters,"Name","conv3_path"+num2str(ii),"Padding","same")
        additionLayer(2,"Name","addition1_path"+num2str(ii))
        reluLayer("Name","relu3_path"+num2str(ii))
        convolution2dLayer([3 3],num_filters,"Name","conv4_path"+num2str(ii),"Padding","same")
        reluLayer("Name","relu4_path"+num2str(ii))
        convolution2dLayer([3 3],num_filters,"Name","conv5_path"+num2str(ii),"Padding","same")
        additionLayer(2,"Name","addition2_path"+num2str(ii))
        reluLayer("Name","relu5_path"+num2str(ii))
        convolution2dLayer([3 3],num_filters,"Name","conv6_path"+num2str(ii),"Padding","same")
        reluLayer("Name","relu6_path"+num2str(ii))
        convolution2dLayer([3 3],num_filters,"Name","conv7_path"+num2str(ii),"Padding","same")
        additionLayer(2,"Name","addition3_path"+num2str(ii))
        reluLayer("Name","relu7_path"+num2str(ii))
        convolution2dLayer([3 3],num_filters,"Name","conv8_path"+num2str(ii),"Padding","same")
        reluLayer("Name","relu8_path"+num2str(ii))
        convolution2dLayer([3 3],num_filters,"Name","conv9_path"+num2str(ii),"Padding","same")
        additionLayer(2,"Name","addition4_path"+num2str(ii))
        reluLayer("Name","relu9_path"+num2str(ii))
        additionLayer(2,"Name","addition5_path"+num2str(ii))];
    lgraph = addLayers(lgraph,tempLayers);

    if ii==1
        tempLayers = [    
            convolution2dLayer([3 3],num_filters,"Name","conv10_path1","Padding","same")
            reluLayer("Name","relu10_path1")];
    else
        tempLayers = [
            maxUnpooling2dLayer("Name","maxunpool1_path"+num2str(ii))
            convolution2dLayer([3 3],num_filters,"Name","conv10_path"+num2str(ii),"Padding","same")
            reluLayer("Name","relu10_path"+num2str(ii))];
    end
    lgraph = addLayers(lgraph,tempLayers);

end
tempLayers = [
    depthConcatenationLayer(num_paths,"Name","depthcat")
    convolution2dLayer([3 3],NoOut,"Name","conv1_out","Padding","same")
    reluLayer("Name","relu1_out")
    regressionLayer("Name","regressionoutput")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

%create connections
for ii=1:num_paths
    lgraph = connectLayers(lgraph,"imageinput","conv1_path"+num2str(ii));
    if ii==1
        lgraph = connectLayers(lgraph,"relu_1_path1","conv2_path1");
        lgraph = connectLayers(lgraph,"relu_1_path1","addition1_path1/in2");
        lgraph = connectLayers(lgraph,"relu_1_path1","addition5_path1/in2");
        lgraph = connectLayers(lgraph,"addition5_path1","conv10_path1");
    else
        lgraph = connectLayers(lgraph,"maxpoolForUnpool1_path"+num2str(ii)+"/out","conv2_path"+num2str(ii));
        lgraph = connectLayers(lgraph,"maxpoolForUnpool1_path"+num2str(ii)+"/out","addition1_path"+num2str(ii)+"/in2");
        lgraph = connectLayers(lgraph,"maxpoolForUnpool1_path"+num2str(ii)+"/out","addition5_path"+num2str(ii)+"/in2");
        lgraph = connectLayers(lgraph,"maxpoolForUnpool1_path"+num2str(ii)+"/indices","maxunpool1_path"+num2str(ii)+"/indices");
        lgraph = connectLayers(lgraph,"maxpoolForUnpool1_path"+num2str(ii)+"/size","maxunpool1_path"+num2str(ii)+"/size");
        lgraph = connectLayers(lgraph,"addition5_path"+num2str(ii),"maxunpool1_path"+num2str(ii)+"/in");
    end

    lgraph = connectLayers(lgraph,"relu3_path"+num2str(ii),"addition2_path"+num2str(ii)+"/in2");
    lgraph = connectLayers(lgraph,"relu5_path"+num2str(ii),"addition3_path"+num2str(ii)+"/in2");
    lgraph = connectLayers(lgraph,"relu7_path"+num2str(ii),"addition4_path"+num2str(ii)+"/in2");
    lgraph = connectLayers(lgraph,"relu10_path"+num2str(ii),"depthcat/in"+num2str(ii));
end

% plot(lgraph)
end