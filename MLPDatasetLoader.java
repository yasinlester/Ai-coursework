package ml.DigitclassifierMLP;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class MLPDatasetLoader {
    
    // This method loads data from a CSV file and returns it as a 2D array of doubles
    public double[][] loadData(String filePath) {
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            int rowCount = 0;

            // This part counts how many rows are in the data-set
            while ((line = br.readLine()) != null) {
                rowCount++; // This counts each line as one row of data to track how many rows are in the file
            
            }

            // Now that we know how many rows there are, we reset the reader to the start of the file
            br.close(); // Close the current BufferedReader to free up the file
            BufferedReader br2 = new BufferedReader(new FileReader(filePath)); // This opens a new BufferedReader for a fresh start
            
            // This assume that the first row determines how many columns we have
            String[] firstRow = br2.readLine().split(","); // This splits the first row by commas to see how many columns it has
            int colCount = firstRow.length; // The number of elements in this split gives us the number of columns
            
            // This sets up a 2D array to store the data 
            double[][] data = new double[rowCount][colCount];

            // This fills in the first row of the data array
            data[0] = new double[colCount]; // This creates a space for the first row in the 2D array
            for (int columnIndex = 0; columnIndex < colCount; columnIndex++) {
                // This convert each value from a string to a double and store it in the first row
                data[0][columnIndex] = Double.parseDouble(firstRow[columnIndex]);
            }

            int rowIndex = 1; // We already filled in row 0, so start from row 1
            while ((line = br2.readLine()) != null) {
                // This splits each line into separate values
                String[] values = line.split(",");
                for (int colIndex = 0; colIndex < values.length; colIndex++) {
                    // This converts each value from a string to a double and store it in the corresponding row and column
                    data[rowIndex][colIndex] = Double.parseDouble(values[colIndex]);
                }
                rowIndex++; //This moves to the next row in the 2D array
            }
            br2.close(); // This closes the second BufferedReader when done
            return data; // This returns the complete 2D array of data

        } catch (IOException e) {
            // If something goes wrong while reading the file, print the error message
            e.printStackTrace(); 
            return null; // This return null to signal that something went wrong
        }
    }
}

