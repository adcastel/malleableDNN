#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "omp.h"
//#include "cblas.h"
#include "mkl.h"

#define INPUT -1
#define FC 0
#define CONV 1
#define APOOL 2
#define MPOOL 3

#define START_TIMER_FP(f,s,l) f[s][l] = omp_get_wtime();
#define END_TIMER_FP(s,l)         fp_comp_timer[s][l] = omp_get_wtime() - fp_comp_timer[s][l];\
        m = nneurons[l];\
        fp_comp_gflops[s][l] = (2.0*m*n*k/fp_comp_timer[s][l])/(1.0e+9);\
        fp_comp_gflops_per_thread[s][l] = fp_comp_gflops[s][l]/(1.0*OMP_NUM_THREADS*procs[l]); 


/* Model features */

#define NUM_STEPS  10  // Steps of the simulation
#define BATCH_SIZE  64 // Batch size

///////////////////////////// PARSER ////////////////////////////////
#define MAX_LEN 1024


const char* getfield(char* line, int num){
  const char* tok;
  for (tok = strtok(line, ";");
    tok && *tok;
    tok = strtok(NULL, ";\n"))
    if (!--num)
      return tok;
  return NULL;
}

int getfield_int(char* line, int num){
  char *l= strdup(line);
  const char *field= getfield(l, num); 
  if (field != NULL) {
    return atoi(field); 
  }
  free(l);
  return 0;
}

double getfield_double(char* line, int num){
  char *l= strdup(line);
  const char *field= getfield(l, num); 
  if (field != NULL) {
    return atof(field); 
  }
  free(l);
  return 0;
}

int count_layers(FILE *fp){
  int num_layers= 0;
  while(!feof(fp))
  {
    char ch = fgetc(fp);
    if(ch == '\n')
    {
      num_layers++;
    }
  }
  return num_layers;
}

//void read_model(FILE *fp, param_t *p, model_t *m){


/* helper functions */
int problem_size(int elements, int nprocs, int rank);

/* Computation functions */
void FC_gemm_fp(int m, int n, int k, float * A, int lda,
        float * B, int ldb, float * C, int ldc);
void CONV_fp(int l, int K, int B, int H, int W, int KH, int KW, int C,
        float * I, float * IP, float * O, float * F, double * time);

/* Communication functions */

int main(int argc, char * argv []) {

    int rank, size, i, s, l;
    double alpha = 1.0, beta = 0.0;
    
    
    if (argc < 2){
      perror("Usage: ./dnn model.csv\n");
      exit(-1);
    }



    FILE *fp_model, *fp_results;
    int aux, j;
    char auxstr[200], auxstr2[200], *token, *str;
    printf("Model: %s\n", argv[1]);
    fp_model= fopen(argv[1], "r");
    //printf("layers: %d\n",count_layers(fp_model));
    int NUM_LAYERS = count_layers(fp_model)-1; //we discard the info line
    
    fclose(fp_model);
     int * type = malloc(sizeof(int)*NUM_LAYERS);
     int * nneurons = malloc(sizeof(int)*NUM_LAYERS);
     int * min_size = malloc(sizeof(int)*NUM_LAYERS);
     int * image_size = malloc(sizeof(int)*NUM_LAYERS);
     int * nkernels = malloc(sizeof(int)*NUM_LAYERS);
     int * channels = malloc(sizeof(int)*NUM_LAYERS);
     int * kwidth = malloc(sizeof(int)*NUM_LAYERS);
     int * kheight = malloc(sizeof(int)*NUM_LAYERS);
     int * vstrides = malloc(sizeof(int)*NUM_LAYERS);
     int * hstrides = malloc(sizeof(int)*NUM_LAYERS);
     int * procs = malloc(sizeof(int)*NUM_LAYERS);
    char line[MAX_LEN]; 
    i = 0;
    int minsfc = 512;
    int minsconv = 8;
    //fclose(fp_model);
    
    fp_model= fopen(argv[1], "r");
    fgets(line, MAX_LEN, fp_model);
    while(fgets(line, MAX_LEN, fp_model)){
      char* tmp = strdup(line);
      const char* typel = getfield(tmp, 2); 
      nneurons[i]  = getfield_int(line, 3) * getfield_int(line, 4) * getfield_int(line, 5); // width * height * channels
      image_size[i]= getfield_int(line, 3);
      channels[i]  = getfield_int(line, 5);
      kwidth[i]    = getfield_int(line, 6);
      kheight[i]   = getfield_int(line, 7);
      hstrides[i]  = getfield_int(line, 8);
      vstrides[i]  = getfield_int(line, 9);
      procs[i]     = getfield_int(line, 10);

    if ( !strcmp(typel, "input") ){ 
    	type[i] = INPUT; min_size[i]= 0;           nkernels[i]= 0; 
    }
    else if ( !strcmp(typel, "fc") ){ 
    	type[i] = FC;    min_size[i]= minsfc;   nkernels[i]= 0; 
	    channels[i]= 1;  kwidth[i]= 0;   kheight[i]= 0;   image_size[i] = 0; 
	  }
    else if ( !strcmp(typel, "conv") ){ 
    	type[i] = CONV;  min_size[i]= minsconv; nkernels[i]= channels[i];
    } 
    else if ( !strcmp(typel, "apool") ){ 
    	type[i] = APOOL; min_size[i]= minsconv; nkernels[i]= 0; 
    }
    else if ( !strcmp(typel, "mpool") ){ 
    	type[i] = MPOOL; min_size[i]= minsconv; nkernels[i]= 0; 
    }
    if(rank == 0)
      printf("type %d, neurons %d, image_size %d, channels %d, kwidth %d, kheight %d, hstrides %d, vstrides %d,  procs %d\n",type[i],nneurons[i] ,image_size[i],channels[i],kwidth[i],kheight[i],hstrides[i],vstrides[i],procs[i]);
      i++;
    }
    fclose(fp_model);

#ifdef TIMER
    double step_timer[NUM_STEPS];
    double * fp_comp_timer[NUM_STEPS];
    double * fp_im2col_timer[NUM_STEPS];
    double * fp_comp_gflops[NUM_STEPS];
    double * fp_comp_gflops_per_thread[NUM_STEPS];
    
    for (i = 0; i < NUM_STEPS; i++){
        fp_comp_timer[i] = (double *) malloc(sizeof (double) * NUM_LAYERS);
        fp_im2col_timer[i] = (double *) malloc(sizeof (double) * NUM_LAYERS);
        fp_comp_gflops[i] = (double *) malloc(sizeof (double) * NUM_LAYERS);
        fp_comp_gflops_per_thread[i] = (double *) malloc(sizeof (double) * NUM_LAYERS);
    }
#endif
    /* Model */


    const char* env = getenv("OMP_NUM_THREADS");
    int OMP_NUM_THREADS = (env != NULL) ? atoi(env) : 1;


    
    /* We first calculate the max size of matrix A, B and C so we only 
     * allocate once and reuse them for all the execution */

    size_t max_size_fc = 0;
    size_t max_size_conv = 0;
    for (l = 1; l < NUM_LAYERS; l++) {
        if (type[l] == FC) {
            if ((1.0 * nneurons[l] * nneurons[l - 1]) > max_size_fc) {
                max_size_fc = (1.0 * nneurons[l] * nneurons[l - 1]);
            }
        } else {
            if ((nneurons[l] + nkernels[l] * kwidth[l] * kheight[l]) > max_size_conv) {
                max_size_conv = (nneurons[l] + nkernels[l] * kwidth[l] * kheight[l]);
            }
        }
    }

    /* This matrices are for FC layers */
    float * matrix_A = malloc(max_size_fc * sizeof ( float));
    float * matrix_B = malloc(max_size_fc * sizeof ( float));
    float * matrix_C = malloc(max_size_fc * sizeof ( float));

    size_t max_i = channels[0] * BATCH_SIZE * image_size[0] * image_size[0];
    size_t max_ip = channels[0] * kwidth[0] * kheight[0] * BATCH_SIZE * image_size[0] * image_size[0];
    size_t max_o = nkernels[0] * BATCH_SIZE * image_size[0] * image_size[0];
    size_t max_f = nkernels[0] * channels[0] * kwidth[0] * kheight[0];
    for (l = 1; l < NUM_LAYERS; l++) {
        if (type[l] == CONV) {
            size_t mi = channels[l] * BATCH_SIZE * image_size[l] * image_size[l];
            size_t mip = channels[l] * kwidth[l] * kheight[l] * BATCH_SIZE * image_size[l] * image_size[l];
            size_t mo = nkernels[l] * BATCH_SIZE * image_size[l] * image_size[l];
            size_t mf = nkernels[l] * channels[l] * kwidth[l] * kheight[l];
            if (mi > max_i) {
                max_i = mi;
            }
            if (mip > max_ip) {
                max_ip = mip;
            }
            if (mo > max_o) {
                max_o = mo;
            }
            if (mf > max_f) {
                max_f = mf;
            }
        }
    }
    float * conv_i = malloc(max_i * sizeof (float));
    float * conv_ip = malloc(max_ip * sizeof (float));
    float * conv_o = malloc(max_o * sizeof (float));
    float * conv_f = malloc(max_f * sizeof (float)); 
    /* The maximum size of the matrix model fits with the one of the matrix */
    size_t model_size = (max_size_fc > max_size_conv) ? max_size_fc : max_size_conv; //This is the maximum size of the model layers
    float * model = malloc(model_size * sizeof (float));
    /* The data size is equal to the neurons in the first layer per the batch size */
    size_t data_size = nneurons[0] * BATCH_SIZE;
    float * data = malloc(data_size * sizeof (float));




    /* Warm-up zone */
    int omp_warm;
#pragma omp parallel
    {
        omp_warm = omp_get_thread_num();
    }
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
            200, 200, 200, 1,
            matrix_A, 200, matrix_B, 200, 0, matrix_C, 200);

        
        for (s = 0; s < NUM_STEPS; s++) {
#ifdef PROGRESS
            //printf("Starting Step %d\n", s);
#endif
#ifdef TIMER
            step_timer[s] = omp_get_wtime();
#endif
            //Forward pass
            for (l = 1; l < NUM_LAYERS; l++) {
                //printf("FP layer %d ",l);
                        if (type[l] == FC) { //FC
                		//printf("FC \n");
                            int m = nneurons[l]; //nneurons[l]/procs[l];//antes /size
                            int n = BATCH_SIZE;
                            int k = nneurons[l - 1]; //We need to reshape if the previous one was CONV
                            int lda = m;
                            int ldb = k;
                            int ldc = m;
#ifdef TIMER
                            fp_comp_timer[s][l] = omp_get_wtime();
#endif
                            FC_gemm_fp(m, n, k, matrix_A, lda, matrix_B, ldb, matrix_C, ldc);
#ifdef TIMER
                            fp_comp_timer[s][l] = omp_get_wtime() - fp_comp_timer[s][l];
                            m = nneurons[l];
                            fp_comp_gflops[s][l] = (2.0 * m * n * k / fp_comp_timer[s][l]) / (1.0e+9);
                            fp_comp_gflops_per_thread[s][l] = fp_comp_gflops[s][l] / (1.0 * OMP_NUM_THREADS);
#endif

                        } else { //conv
               		    //printf("CONV \n");
                            int num_kernels = nkernels[l]; //nneurons[l]/procs[l];//antes /size
                            int b = BATCH_SIZE;
                            int h = image_size[l - 1];
                            int w = image_size[l - 1];
                            int c = channels[l - 1];
                            int kh = kheight[l];
                            int kw = kheight[l];
#ifdef TIMER
                            fp_comp_timer[s][l] = omp_get_wtime();
#endif
                            CONV_fp(l, num_kernels, b, h, w, kh, kw, c,
                                    conv_i, conv_ip, conv_o, conv_f, &fp_im2col_timer[s][l]);

#ifdef TIMER
                            fp_comp_timer[s][l] = omp_get_wtime() - fp_comp_timer[s][l];
                            int m = nkernels[l];
                            int n = b * h * w;
                            int k = c * kh *kw;
                            fp_comp_gflops[s][l] = (2.0 * m * n * k / fp_comp_timer[s][l]) / (1.0e+9);
                            fp_comp_gflops_per_thread[s][l] = fp_comp_gflops[s][l] / (1.0 * OMP_NUM_THREADS);
#endif
                        }
                    }

#ifdef TIMER
            step_timer[s] = omp_get_wtime() - step_timer[s];
#endif
        } //steps
#ifdef TIMER
#ifndef SUMMARY
            double total_time = 0.0;
            for (s = 0; s < NUM_STEPS; s++) {
                printf("STEP %d\nTime %f\n", s, step_timer[s]);
                printf("\t **** FP ****\n");
                for (l = 1; l < NUM_LAYERS; l++) {
                    printf("\t Layer %d (type %s)\n", l, (type[l] == CONV) ? "Conv" : "FC");
                    printf("\t\t FP Computation time %f", fp_comp_timer[s][l]);
                    if (type[l] == CONV) printf(" (im2col = %f)", fp_im2col_timer[s][l]);
                    printf(" | GFlops %f (cores %d) GFlops/core %f (cores %d)\n", fp_comp_gflops[s][l], OMP_NUM_THREADS, fp_comp_gflops_per_thread[s][l], OMP_NUM_THREADS);
                }
                total_time += step_timer[s];
            }
            printf("Time per step = %f\n", total_time / NUM_STEPS);
#else

            double  total_time_r[NUM_LAYERS], total_time_fp[NUM_LAYERS], total_time_comp_fp[NUM_LAYERS]; 
            
            for (l = 1; l < NUM_LAYERS; l++) {
                total_time_r[l] = 0;
                total_time_fp[l] = 0;
                total_time_comp_fp[l] = 0;

            }
            for (l = 1; l < NUM_LAYERS; l++) {
            for (s = 1; s < NUM_STEPS; s++) {
                    total_time_fp[l] += fp_comp_timer[s][l];
                    total_time_comp_fp[l] += fp_comp_timer[s][l];
}
                    total_time_r[l] += total_time_fp[l];
                }
            double tt = 0.0;
            printf("#layer #threads total_time \n");
            for (l = 1; l < NUM_LAYERS; l++) {
		tt+=total_time_fp[l];
                printf("%d %d %f\n", l, OMP_NUM_THREADS, total_time_fp[l] / (NUM_STEPS - 1)); 
            }
            printf("Total %f \n", tt/ (NUM_STEPS - 1));

#endif
#endif    


    free(matrix_A);
    free(matrix_B);
    free(matrix_C);
        free(model);
      free(data);
    return 0;
}

void FC_gemm_fp(int m, int n, int k, float * A, int lda, float * B, int ldb, float * C, int ldc) {
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
           m, n, k, 1,
            A, lda, B, ldb, 0, C, ldc);
    //printf("FP  FC  GEMM m(%d) : n(%d) : k(%d)\n", m, n, k);
}


void CONV_fp(int l, int K, int B, int H, int W, int KH, int KW, int C, float * I, float * IP, float * O, float * F, double * time) {

    // B batch size
    // Input image of size H x W, with C channels
    // K Kernels of size KH x KW each, with C channels
    /*
    float I[C][B][H][W];          // Input: C x (B H W)
    float IP[C][KW][KH][B][H][W]; // Patched input: (C K_H K_W) x (B H W)
    float O[K][B][H][W];          // Output: K x (B H W)
    float F[K][C][KH][KW];        // Filter: K x (C K_H K_W)
     */

    int b, h, w, kh, kw, c;
#ifdef TIMER
    *time = omp_get_wtime();
#endif
    // Im2col: I -> IP

    int kk1 = KH * KW * B * H*W;
    int kk2 = KW * B * H*W;
    int kk3 = B * H*W;
    int kk4 = H*W;
    int kk5 = B * (H + KH)*(W + KW);
    int kk6 = (H + KH)*(W + KW);
    int kk7 = (W + KW);
    int jk1, ik1, ik2, jk2, jk3, jk4, ik3, ik4, ik5;
//printf("Antes de im2col\n");
#pragma omp parallel for private(b,h,w,kh,kw,ik1,ik2,ik3,ik4,ik5,jk1,jk2,jk3,jk4)
    for (c = 0; c < C; c++) {
        ik1 = c*kk1;
        jk1 = c*kk5;
        for (b = 0; b < B; b++) {
            ik2 = ik1 + b*kk4;
            jk2 = jk1 + b*kk6;
            for (kh = 0; kh < KH; kh++) {
                ik3 = ik2 + kh*kk2;
                jk3 = jk2 + (h + kh) * kk7;
                for (kw = 0; kw < KW; kw++) {
                    ik4 = ik3 + kw * kk3;
                    jk4 = jk3 + kw;
                    for (h = 0; h < H; h++) {
                        ik5 = ik4 + h*W;
                        for (w = 0; w < W; w++)
                            IP[ ik5 + w ] = I[ jk4 + w ];
                    }
                }
            }
        }
    }
#ifdef TIMER
    *time = omp_get_wtime() - *time;
    /* char processor_name[MPI_MAX_PROCESSOR_NAME];
       int name_len;
       MPI_Get_processor_name(processor_name, &name_len);
     double a = *time;    
     printf("%s Layer %d FP space=%f time=%f  GB/s = %f\n",processor_name,l,C*B*H*W*KH*KW*2.0,a,(C*B*H*W*KH*KW*2.0*4.0/1000000000.0)/a);
     */
#endif
    // Gemm
//printf("Antes de gemm\n");
    int m = K;
    int n = B * H*W;
    int k = C * KH*KW;
    int lda = m;
    int ldb = k;
    int ldc = m;

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
            m, n, k, 1,
            F, lda, IP, ldb, 0, O, ldc);
//printf("despues de gemm\n");

    //printf("FP CONV GEMM m(%d) : n(%d)=b(%d).h(%d).w(%d) : k(%d)=c(%d).kh(%d).kw(%d)\n", m, n, B, H, W, k, C, KH, KW);

    /*free(miI);
    free(miIP);
    free(miO);
    free(miF);*/
    /*GEMM -> O = F * IP

            No transpose, No transpose

            alpha = 1.0

            beta  = 0.0
            m = K, n = (B H W), k = (C K_H K_W) 
     */
}


int problem_size(int elements, int nprocs, int rank) {
    int part = elements / nprocs;
    int rest = elements % nprocs;
    if (rest > 0) {
        if (rank < rest) {
            part++;
        }
    }
    return part;
}
