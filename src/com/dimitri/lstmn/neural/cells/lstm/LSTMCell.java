package com.dimitri.lstmn.neural.cells.lstm;

import com.dimitri.lstmn.neural.cells.Cell;
import com.dimitri.lstmn.neural.cells.Connection;
import com.dimitri.lstmn.neural.cells.ProcessorCell;
import com.dimitri.lstmn.neural.layers.Layer;

public class LSTMCell extends Cell{

    private Connection[] connection;
    private Connection[] recurrentConnection;

    private Gate inputGate;
    private Gate forgetGate;
    private Gate outputGate;

    private double forgetGateMemory;

    private double gradientInput;
    private double gradientInputGate;
    private double gradientForgetGate;
    private double gradientOutputGate;
    private double gradientState;
    private double gradientOuter;


    public LSTMCell(Layer layer, int cellIndex, int inputAmount) {
        super(layer, cellIndex);

        this.connection = new Connection[inputAmount];
        this.recurrentConnection = new Connection[layer.getCell().length];
        for (int i = 0; i < connection.length; i++) {
            connection[i] = new Connection(ProcessorCell.getRandomWeight());
        }
        for (int i = 0; i < recurrentConnection.length; i++) {
            recurrentConnection[i] = new Connection(ProcessorCell.getRandomWeight());
        }

        this.inputGate = new Gate(inputAmount, layer.getCell().length);
        this.forgetGate = new Gate(inputAmount, layer.getCell().length);
        this.outputGate = new Gate(inputAmount, layer.getCell().length);
        this.forgetGateMemory = 0;
    }

    public void feedForward(){
        double solution = 0;
        double sum = 0;
        //new Data
        Layer previousLayer = getLayer().getNet().getLayer(getLayer().getLayerIndex()-1);

        for (int i = 0; i < previousLayer.getCell().length; i++) {
            sum += previousLayer.getCell(i).getOutput() * connection[i].getWeight();
        }

        //recurrent Data
        for (int i = 0; i < getLayer().getCell().length; i++) {
            sum += getLayer().getCell(i).getOutput() * recurrentConnection[i].getWeight();
        }

        sum += 1*connection[connection.length-1].getWeight();

        //Normalize
        solution = ProcessorCell.tanh(sum);

        //inputGate
        double inputGateKeeper = inputGate.feedForward(previousLayer, getLayer());

        //multiply
        solution = solution * inputGateKeeper;

        //forgetGate
        double forgetGateKeeper = forgetGate.feedForward(previousLayer, getLayer());;

        //multiply with previous timestep
        forgetGateKeeper = forgetGateMemory * forgetGateKeeper;

        //sum
        solution += forgetGateKeeper;

        //set new memory
        forgetGateMemory = solution;

        //normalize
        solution = ProcessorCell.tanh(solution);

        //outputGate
        double outputGateKeeper = outputGate.feedForward(previousLayer, getLayer());

        //multiply
        solution = solution * outputGateKeeper;

        //setOutput
        setOutput(solution);
    }

    public void calcOutputGradient(double targetOutput){
        double delta = targetOutput - getOutput();


        gradient = delta * ProcessorCell.derrivativeTanh(getOutput());
    }

    public void calcHiddenGradient(){
        double sum = 0;

        for (int i = 0; i < getLayer().getNet().getLayer(getLayer().getLayerIndex()+1).getCell().length; i++) {
            sum += ((HiddenCell)getLayer().getNet().getLayer(getLayer().getLayerIndex()+1).getCell(i)).getConnection(getCellIndex()).getWeight()*((HiddenCell)getLayer().getNet().getLayer(getLayer().getLayerIndex()+1).getCell(i)).getGradient();
        }
        double dow = sum;
        gradient = dow * ProcessorCell.derrivativeTanh(getOutput());
    }



}
