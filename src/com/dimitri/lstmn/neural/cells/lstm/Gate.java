package com.dimitri.lstmn.neural.cells.lstm;

import com.dimitri.lstmn.neural.cells.Connection;
import com.dimitri.lstmn.neural.cells.ProcessorCell;
import com.dimitri.lstmn.neural.layers.Layer;

public class Gate {

    private Connection[] connection;
    private Connection[] recurrentConnection;

    public Gate(int inputAmount, int recurrentAmount){
        this.connection = new Connection[inputAmount];
        this.recurrentConnection = new Connection[recurrentAmount];
        for (int i = 0; i < connection.length; i++) {
            connection[i] = new Connection(ProcessorCell.getRandomWeight());
        }
        for (int i = 0; i < recurrentConnection.length; i++) {
            recurrentConnection[i] = new Connection(ProcessorCell.getRandomWeight());
        }
    }

    public double feedForward(Layer prev, Layer current){
        double sum = 0;
        //new Data
        for (int i = 0; i < prev.getCell().length; i++) {
            sum += prev.getCell(i).getOutput() * connection[i].getWeight();
        }

        //recurrent Data
        for (int i = 0; i < current.getCell().length; i++) {
            sum += current.getCell(i).getOutput() * recurrentConnection[i].getWeight();
        }

        sum += 1*connection[connection.length-1].getWeight();

        return ProcessorCell.sigmoid(sum);
    }

}
