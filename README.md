# Graph-Signal-Processing-datausa.io
This code is about GSP with signals from datausa.io . Some functions come from gspy library (https://github.com/gboaviagem/GSPy) . 

You can find csv files in datausa.io that have a standard format. Using this format is possible to make a graph plot directly from this file. We have 3 plots. The first one is about a graph (V,E), V is about the counties, so each node has a location from a county and the edges were weighted using euclidean distance (exp(-dist²)). Note that all the edges are bidirectional and the adjacency matrix were built connecting the closer three nodes. The second one is about a GFT plot using Total Variation for frequency analysis. The last one is about a filtered signal (low pass filter). More details can be seen in the code.
