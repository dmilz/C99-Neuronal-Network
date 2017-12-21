#include "nn.h"

typedef struct Dataset
{
  double in[1000];
  double out[1000];
  unsigned int length;
} Dataset;

Dataset getEmptyDataset()
{
    Dataset data;
    memset(data.in,0,sizeof(data.in));
    memset(data.out,0,sizeof(data.out));
    data.length = 0;
    return data;
}

void initDataset(Dataset * data)
{
    memset(data->in,0,sizeof(data->in));
    memset(data->out,0,sizeof(data->out));
    data->length = 0;
}

void readData(Dataset * tr, Dataset * ev)
{
    // read training data ending with 0,0,0 input
    while (scanf("%lf,%lf", &tr->in[tr->length], &tr->out[tr->length]) > 1) {
        if ((tr->in[tr->length] == 0) && (tr->out[tr->length] == 0)) break;

        tr->length++;
        if (tr->length >= 1000) break;
    }

    // read evaluation data
    while(scanf("%lf",&ev->in[ev->length]) >= 1) {
        ev->length++;
        if(ev->length >= 1000) break;
    }
}

void normalize(Dataset ** datasets, int nData, double *maximal_value)
{
    double * maxval = calloc(1, sizeof(double));

    for(int j=0; j<nData; j++) {
        Dataset * data = datasets[j];
        for (unsigned int i = 0; i < data->length; i++) {
            *maxval = (data->in[i] > *maxval ? data->in[i] : *maxval);
        }
    }
    for(int j=0; j<nData; j++) {
        Dataset *data = datasets[j];
        for (unsigned int i = 0; i < data->length; i++) {
            data->in[i] = data->in[i] / fabs(*maxval);
        }
    }

    if (maximal_value != NULL) {
        *maximal_value = *maxval;
    }

    free(maxval);
}

int main(int argc, char * argv[])
{
#ifdef DEBUG
    freopen("example_in.txt","r",stdin);
#endif

    Dataset training_data = getEmptyDataset();
    Dataset evaluation_data = getEmptyDataset();
    readData(&training_data, &evaluation_data);
    Dataset * datasets[2] = {&training_data, &evaluation_data};
    double maxValue = 0.0;
    normalize(datasets,2,&maxValue);

    Network n;
    if (!loadNetwork(&n)) {
        const int nInputs = 1;
        const int nHiddenLayer = 5;
        const int hiddenLayer[] = {10, 10, 10, 10, 10};
        const int nOutputs = 1;
        srand((unsigned int) time(NULL));
        initNetwork(&n, nInputs, nHiddenLayer, (int *) hiddenLayer, nOutputs);
    }

    Optimizer o;
    o.costFcn = meanSqauareError;
    o.eta = 0.001;
    o.etaDecay = 1.0;
    o.epochs = 10000;
    o.training_data = &training_data;

    optimize(&n,&o);

    saveNetwork(&n);

    run(&n,&evaluation_data);

    { // BEGIN print results
        for (int i=0; i<evaluation_data.length; i++)
            printf("%lf\n",evaluation_data.out[i]);
    } // END print results

    freeNetwork(&n);
    return 0;
}
