#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

typedef double (*actFun)(double x);
typedef double (*costFun)(const double * realOut, const double * targetOut, int size);

typedef struct Neuron
{
	int nIn;
	double * in;
	double net;
	double out;
	double * weights;
	double delta;
	actFun actFcn;
	actFun dActFcn;
    int nFanOut;
    struct Neuron ** folNeuron;
} Neuron;

typedef struct Layer
{
	int nNeurons;
	Neuron * n;
    int nIn;
    double * in;
    double * out;
} Layer;

typedef struct Network
{
	int nIn;
	double * in;
	int nOut;
	double * out;
	int nHiddenLayer;
	Layer * hiddenLayer;
	Layer outputLayer;
} Network;

typedef struct Optimizer
{
    costFun costFcn;
    double eta; // learning rate
    double etaDecay;
    int epochs;
    Dataset * training_data;
} Optimizer;

enum ACTIVATIONFUNCTOIN
{
	actLinear, actSigmoidal
};

static double linear(double x) { return x; }
static double dLinear(double x) { return 1.0; }
static double tanH(double x) { return tanh(x); }
static double dTanH(double x) { return 1-tanh(x)*tanh(x); }
static double meanSqauareError(const double * realOut, const double * targetOut, int size)
{
    double d = 0.0;
    for (int i=0; i<size; i++)
        d += (realOut[i]*realOut[i] - targetOut[i]*targetOut[i]);
    return fabs(d);
}


void initNeuron(Neuron * n, int nIn, enum ACTIVATIONFUNCTOIN act, Layer * followingLayer)
{
    const double maxWeightInitialization = 0.01;
    int fanOut = 0;
    if (followingLayer != NULL) {
        fanOut = followingLayer->nNeurons;
    }

	n->nIn = nIn;
	n->in = calloc((size_t) n->nIn, sizeof(double));
	n->net = 0.0;
	n->out = 0.0;
	n->weights = calloc((size_t) (n->nIn+1), sizeof(double));
    // Randomly initialize weights
    for (int i=0; i<n->nIn +1; i++) {
        n->weights[i] = (((double) rand()) / RAND_MAX) * maxWeightInitialization - (maxWeightInitialization/2.0);
    }

	n->delta = 0.0;
	if (act == actLinear) {
		n->actFcn = &linear;
		n->dActFcn = &dLinear;
	} else {
		n->actFcn = &tanH;
		n->dActFcn = &dTanH;
	}

    n->nFanOut = fanOut;
    n->folNeuron = (n->nFanOut ? calloc((size_t) n->nFanOut, sizeof(Neuron*)) : NULL);
    for (int i=0; i<fanOut; i++) {
        n->folNeuron[i] = &followingLayer->n[i];
    }
}

void freeNeuron(Neuron * n)
{
	free(n->in);
	free(n->weights);
}

void initLayer(Layer * l, int nNeurons, int nIn, enum ACTIVATIONFUNCTOIN act, Layer * followingLayer)
{
    l->nNeurons = nNeurons;
    l->n = calloc((size_t) l->nNeurons, sizeof(Neuron));
    for (int i = 0; i < l->nNeurons; i++) {
        initNeuron(&l->n[i], nIn, act, followingLayer);
    }

    l->nIn = nIn;
    l->in = calloc((size_t) l->nIn, sizeof(double));
    l->out = calloc((size_t) l->nNeurons, sizeof(double));
}

void freeLayer(Layer * l)
{
    for (int i = 0; i < l->nNeurons; i++) {
        freeNeuron(&l->n[i]);
    }
    free(l->n);
    free(l->in);
    free(l->out);
}

void initNetwork(Network * n, int nIn, int nHiddenLayer, const int * hiddenLayerSize, int nOut)
{
    n->nIn = nIn;
    n->in = calloc((size_t) n->nIn, sizeof(double));
    n->nOut = nOut;
    n->out = calloc((size_t) n->nOut, sizeof(double));

    n->nHiddenLayer = nHiddenLayer;

    int tmp_nNeurons = n->nOut;
    int tmp_nIn = (n->nHiddenLayer == 0 ? n->nIn : hiddenLayerSize[n->nHiddenLayer-1]);
    Layer * lastLayer = NULL; // For giving every layer its following layer

    initLayer(&n->outputLayer, tmp_nNeurons, tmp_nIn, actLinear, lastLayer);
    n->hiddenLayer = (n->nHiddenLayer ? calloc((size_t) n->nHiddenLayer, sizeof(Layer)) : NULL);
    if (nHiddenLayer == 0) {
        n->hiddenLayer = NULL;
    } else {
        for(int i = n->nHiddenLayer-1; i >= 0; i--) {
            tmp_nNeurons = hiddenLayerSize[i];
            tmp_nIn = (i == 0 ? n->nIn : hiddenLayerSize[i - 1]);
            lastLayer = (i==n->nHiddenLayer-1 ? &n->outputLayer : &n->hiddenLayer[i+1]);

            initLayer(&n->hiddenLayer[i], tmp_nNeurons, tmp_nIn, actSigmoidal, lastLayer);
        }
    }
}

void freeNetwork(Network * n)
{
    free(n->in);
    free(n->out);
    freeLayer(&n->outputLayer);
    for(int i=0; i < n->nHiddenLayer; i++) {
        freeLayer(&n->hiddenLayer[i]);
    }
    free(n->hiddenLayer);
}

int readNetworkFromFile(Network * n, FILE * in)
{
    int inputs, hidden_layers, outputs;
    int * hiddenLayerSize;

    int rc = fscanf(in, "%d, %d, ", &inputs, &hidden_layers);
    if (rc < 2) {
        perror("fscanf");
        return 0;
    }
    hiddenLayerSize = calloc((size_t) hidden_layers, sizeof(int));

    for (int i=0; i<hidden_layers; i++) {
        rc = fscanf(in, "%d, ", &hiddenLayerSize[i]);
        if (rc < 1) {
            perror("fscanf");
            return 0;
        }
    }

    rc = fscanf(in, "%d\n\n", &outputs);
    if (rc < 1) {
        perror("fscanf");
        return 0;
    }

    initNetwork(n,inputs,hidden_layers,hiddenLayerSize,outputs);

    // Hidden layer
    for (int lyer = 0; lyer < hidden_layers; lyer++) {
        Layer * layer = &n->hiddenLayer[lyer];
        for (int nron = 0; nron < layer->nNeurons; nron++) {
            Neuron * neuron = &layer->n[nron];
            for (int i = 0; i < neuron->nIn; i++) {
                rc = fscanf(in, "%le, ", &neuron->weights[i]);
                if (rc < 1) {
                    perror("fscanf");
                    freeNetwork(n);
                    return 0;
                }
            }
            rc = fscanf(in, "%le\n", &neuron->weights[neuron->nIn]); // Bias
            if (rc < 1) {
                perror("fscanf");
                freeNetwork(n);
                return 0;
            }
        }
        fscanf(in, "\n");
    }

    // Output layer
    for (int nron = 0; nron < n->outputLayer.nNeurons; nron++) {
        Neuron * neuron = &n->outputLayer.n[nron];
        for (int i = 0; i < neuron->nIn; i++) {
            rc = fscanf(in, "%le, ", &neuron->weights[i]);
            if (rc < 1) {
                perror("fscanf");
                freeNetwork(n);
                return 0;
            }
        }
        rc = fscanf(in, "%le\n", &neuron->weights[neuron->nIn]); // Bias
        if (rc < 1) {
            perror("fscanf");
            freeNetwork(n);
            return 0;
        }
    }
    return 1;
}

int loadNetwork(Network * n)
{
    FILE *file = fopen("config", "r");
    if (!file) {
        printf("Could not open file to read!");
        return 0; // Failed
    }

    if (readNetworkFromFile(n, file)) {
#ifdef DEBUG
        printf("Successfully loaded network!");
#endif
        return 1; // Success
    }

    return 0;
}

void writeNetworkToFile(const Network * n, FILE *out)
{
    fprintf(out, "%d, %d, ", n->nIn, n->nHiddenLayer);
    for (int i=0; i<n->nHiddenLayer; i++) {
        fprintf(out, "%d, ", n->hiddenLayer[i].nNeurons);
    }
    fprintf(out, "%d\n\n", n->nOut);


    // Hidden layer
    for (int lyer = 0; lyer < n->nHiddenLayer; lyer++) {
        Layer * layer = &n->hiddenLayer[lyer];
        for (int nron = 0; nron < layer->nNeurons; nron++) {
            Neuron * neuron = &layer->n[nron];
            for (int i = 0; i < neuron->nIn; i++) {
                fprintf(out, "%+.20e, ", neuron->weights[i]);
            }
            fprintf(out, "%+.20e\n", neuron->weights[neuron->nIn]); // Bias
        }
        fprintf(out, "\n");
    }

    // Output layer
    for (int nron = 0; nron < n->outputLayer.nNeurons; nron++) {
        Neuron * neuron = &n->outputLayer.n[nron];
        for (int i = 0; i < neuron->nIn; i++) {
            fprintf(out, "%+.20e, ", neuron->weights[i]);
        }
        fprintf(out, "%+.20e\n", neuron->weights[neuron->nIn]); // Bias
    }
}

int saveNetwork(const Network *n)
{
    FILE *file = fopen("config", "w");
    if (!file) {
        printf("Could not open file to write!");
        return 0; // Failes
    }
    writeNetworkToFile(n, file);
#ifdef DEBUG
    printf("Successfully saved network!");
#endif
    return 1; // Success
}

// out can be NULL
void forwardPassNeuron(Neuron * n, double * in, double * out)
{
    if (in != NULL) {
        memcpy(n->in, in, n->nIn * sizeof(double));
    }

    n->net = n->weights[n->nIn] * (-1.0); // Bias
    for (int i=0; i<n->nIn; i++) {
        n->net += n->weights[i] * n->in[i];
    }

    n->out = n->actFcn(n->net);

    if(out != NULL) {
        *out = n->out;
    }
}

// in and out can be NULL
void forwardPassLayer(Layer * l, double * in, double * out) {
    if (in != NULL) {
        memcpy(l->in, in, l->nIn * sizeof(double));
    }

    for (int i = 0; i < l->nNeurons; i++) {
        forwardPassNeuron(&l->n[i], l->in, &l->out[i]);
    }

    if (out != NULL) {
        memcpy(out, l->out, l->nNeurons * sizeof(double));
    }
}

//out can be NULL
void forwardPassNetwork(Network * n, double * in, double * out)
{
    if (in != NULL) {
        memcpy(n->in, in, n->nIn * sizeof(double));
    }

    double * tmp_i, * tmp_o;
    for (int i=0; i<n->nHiddenLayer; i++) {
        tmp_i = (i == 0 ? in : n->hiddenLayer[i - 1].out);
        tmp_o = (i < (n->nHiddenLayer - 1) ? n->hiddenLayer[i + 1].in : n->outputLayer.in);
        forwardPassLayer(&n->hiddenLayer[i], tmp_i, tmp_o);
    }
    forwardPassLayer(&n->outputLayer, (n->nHiddenLayer ? n->outputLayer.in : in), n->out);

    if (out != NULL) {
        memcpy(out, n->out, n->nOut * sizeof(double));
    }
}

void optimizeBatch(Network * n, costFun costFcn, double learningRate, double * in, const double * targetOut)
{
    double * realOut = calloc((size_t) n->nOut, sizeof(double));

    forwardPassNetwork(n,in,realOut);

    //double error = costFcn(realOut,targetOut,n->nOut);

    { // BEGIN Calculate deltas

        // Delta of output layer
        // d_j = phi'(net_j) * (realOut_j - targetOut_j)
        for (int i=0; i<n->outputLayer.nNeurons; i++) {
            Neuron * neuron = &n->outputLayer.n[i];
            neuron->delta = neuron->dActFcn(neuron->net) * (realOut[i] - targetOut[i]);
        }

        // Delta of hidden layer
        // d_j = phi'(net_j) * sum_k delta_k*w_jk
        for(int i=n->nHiddenLayer-1; i>=0; i--) {
            Layer * layer = &n->hiddenLayer[i];
            for (int j=0; j < layer->nNeurons; j++) {
                Neuron * neuron = &layer->n[i];
                double sum_k = 0.0;
                for (int k=0; k < neuron->nFanOut; k++) {
                    sum_k += neuron->folNeuron[k]->delta * neuron->folNeuron[k]->weights[j];
                }
                neuron->delta = sum_k * neuron->dActFcn(neuron->net);
            }
        }

    } // END Calculate deltas

    { // BEGIN Update weights: dW_j[i] = -eta * delta_j * out_i = -eta * delta_j * in_j[i]

        // Update weights of hidden layer
        for(int lyer=0; lyer < n->nHiddenLayer; lyer++) {
            Layer * layer = &n->hiddenLayer[lyer];
            for (int j=0; j < layer->nNeurons; j++) {
                Neuron * neuron = &layer->n[lyer];
                for (int i=0; i < neuron->nIn+1; i++) {
                    double in_j = (i < neuron->nIn ?  neuron->in[i] : -1.0);
                    neuron->weights[i] = (- learningRate * neuron->delta * in_j);
                }
            }
        }

        // Update weights of output layer
        for (int j=0; j < n->outputLayer.nNeurons; j++) {
            Neuron * neuron = &n->outputLayer.n[j];
            for (int i=0; i < neuron->nIn+1; i++) {
                double in_j = (i < neuron->nIn ?  neuron->in[i] : -1.0);
                neuron->weights[i] = (- learningRate * neuron->delta * in_j);
            }
        }
    } // END Update weights

    free(realOut);
}

void optimize(Network * n, Optimizer * o)
{
    for (int epoch=0; epoch < o->epochs; epoch++) {
        for (int batch=0; batch < o->training_data->length; batch++) {
            optimizeBatch(n, o->costFcn, o->eta, &o->training_data->in[batch], &o->training_data->out[batch]);
        }

        o->eta *= o->etaDecay;
    }
}

void run(Network * n, Dataset * d)
{
    for (int i=0; i<d->length; i++){
        forwardPassNetwork(n, &d->in[i], &d->out[i]);
    }
}
