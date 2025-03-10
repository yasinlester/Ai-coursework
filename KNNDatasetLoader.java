package ml.digitclassifierKnn;

import java.io.*;

public class KNNDatasetLoader {
    
    // This loads the data from a CSV file returns a 2D array of data
    public double[][] loadData(String fileName) {
        try {
            // This first finds where the file is at. If it's not found, the file states an error.
            String filePath = findFile(fileName);
            if (filePath == null) {
                throw new FileNotFoundException("File not found: " + fileName);
            }

            // This counts how many rows are in the file so we know how big our 2D array needs to be
            int rowCount = countRows(filePath);

            // This checks the first line of the file to see how many columns it has
            int columnCount;
            try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
                String firstLine = br.readLine();
                if (firstLine == null) {
                    throw new IOException("File is empty: " + filePath);
                }
                columnCount = firstLine.split(",").length; // Assumes it's a CSV file
            }

            // This creates a 2D array to hold all the data from the file
            double[][] data = new double[rowCount][columnCount];

            // This reads each line of the file and fills in the 2D array with the numbers
            try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
                String line;
                int rowIndex = 0;
                while ((line = br.readLine()) != null && rowIndex < rowCount) {
                    String[] values = line.split(","); // This Splits the line into an array of values
                    fillRow(data, rowIndex, values); // This fills that row of the array with the values
                    rowIndex++;
                }
            }

            // After we have the raw data, we can normalise it so it's ready to be used
            return normalize(data);

        } catch (IOException e) {
            System.err.println("Error reading file: " + e.getMessage()); // If something goes wrong, it prints error
            return new double[0][0]; //It will return an empty array so it doesn't crash everything
        }
    }

    // This takes an array of string values and fills one row of the 2D array with double values
    private void fillRow(double[][] data, int rowIndex, String[] values) {
        for (int index = 0; index < values.length; index++) {
            data[rowIndex][index] = Double.parseDouble(values[index]); // This converts strings to doubles and add them to the row
        }
    }

    // This counts how many rows (lines) are in the file, so we know how many rows to make in the 2D array
    private int countRows(String filePath) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            int rowCount = 0;
            while (br.readLine() != null) {
                rowCount++; // Every line we read is one row of data
            }
            return rowCount; // It will return the number of rows in the file
        }
    }

    // This normalises each feature column in the 2D array so everything's on the same scale
    private double[][] normalize(double[][] data) {
        double[][] normalizedData = new double[data.length][data[0].length];

        //This Normalise all the columns except the last one (assuming it's the label)
        for (int index = 0; index < data[0].length - 1; index++) {
            double min = Double.MAX_VALUE, max = Double.MIN_VALUE;

            // This finds the min and max for the current column
            for (double[] row : data) {
                if (row[index] < min) min = row[index];
                if (row[index] > max) max = row[index];
            }

            // This normalises the numbers in the column using the min and max
            for (int rowIndex = 0; rowIndex < data.length; rowIndex++) {
                normalizedData[rowIndex][index] = (data[rowIndex][index] - min) / (max - min);
            }
        }

        // Copy the label column (the last column) as it is, no changes needed
        for (int index = 0; index < data.length; index++) {
            normalizedData[index][data[0].length - 1] = data[index][data[0].length - 1];
        }

        return normalizedData; // This returns the normalised data
    }

    // This tries to find the file in the current directory or sub-directories
    private String findFile(String fileName) {
        File currentDir = new File(System.getProperty("user.dir")); // This gets the current working directory
        return searchFile(currentDir, fileName); //It starts the search of the file from here
    }

    // This looks for the file with the given name in the directory and its sub-directories
    private String searchFile(File dir, String fileName) {
        for (File file : dir.listFiles()) {
            if (file.isDirectory()) {
                // If this is a directory, search through all its contents, including sub-directories
                String result = searchFile(file, fileName);
                if (result != null) return result; //  If we found the file we're looking for, it returns its absolute path
            } else if (file.getName().equalsIgnoreCase(fileName)) {
                // If the file name matches ignoring case, return its absolute path
                return file.getAbsolutePath();
            }
        }
        return null; // If we didn't find the file, return null
    }
}


