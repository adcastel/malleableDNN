#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <math.h>
#include "omp.h"
//#include "cblas.h"
//#include "mkl.h"
#include "blis.h"

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

#define NUM_STEPS  3  // Steps of the simulation
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

// Esta funcion es para hacer profile, para la version final se llamara
//a my_gemm()
void test_gemm(dim_t m, dim_t n, dim_t k, int max_threads, rntm_t * rnmt ){

    printf("GEMM %dx%dx%d\n", m, n, k);    
    obj_t a, b, c;    
    obj_t alpha, beta;
    num_t dt_a, dt_b, dt_c, dt_alpha, dt_beta;
    //dim_t m, n, k;
    double tini, tend;
    dt_a = BLIS_DOUBLE;    
    dt_b = BLIS_DOUBLE;    
    dt_c = BLIS_DOUBLE;    
    dt_alpha = BLIS_DOUBLE;
    dt_beta = BLIS_DOUBLE; 

    bli_obj_create( dt_alpha, 1, 1, 0, 0, &alpha ); 
    bli_obj_create( dt_beta,  1, 1, 0, 0, &beta );  
                                                
    bli_obj_create( dt_a, m, k, 0, 0, &a );         
    bli_obj_create( dt_b, k, n, 0, 0, &b );         
    bli_obj_create( dt_c, m, n, 0, 0, &c );         
                                                
    bli_randm( &a );                                
    bli_randm( &b );                                
    bli_randm( &c );                                
                                                
                                                
    bli_setsc(  (0.9/1.0), 0.2, &alpha );           
    bli_setsc(  (1.0/1.0), 0.0, &beta );            

    for (int i = 1; i <= max_threads; i++){                                                 
        for (int j = 1; j <= max_threads; j++){                                         
                for (int q = 1; q <= max_threads; q++){                                 
                        for (int l = 1; l <= max_threads; l++){                         
                                if(i*j*q*l > max_threads)                           
                                        continue;                                   
                                printf("%d %d %d %d %d ",i,1,j,q,l);                
                                bli_rntm_set_ways(i,1,j,q,l,rnmt);                 
                                tini = bli_clock();                                 
                                bli_gemm_ex(&alpha, &a, &b, &beta, &c, NULL, rnmt);
                                tend = bli_clock();                                 
                                printf("%f\n", tend-tini);                          
                                                                                    
                        }                                                           
                }                                                                   
        }                                                                           
    }                                                                                   
                                                                                    
    bli_rntm_set_num_threads(max_threads,rnmt);                                        
//bli_rntm_get_ways();                                                              
    tini = bli_clock();                                                                 
    bli_gemm_ex(&alpha, &a, &b, &beta, &c, NULL, rnmt);                                
    tend = bli_clock();                                                                 
    printf("Auto %f\n", tend-tini);                                                     

     bli_obj_free( &alpha );           
       bli_obj_free( &beta );            
                                   
 bli_obj_free( &a );               
 bli_obj_free( &b );               
 bli_obj_free( &c );               

}

void my_gemm(){
    
                                

}

/* helper functions */
int problem_size(int elements, int nprocs, int rank);

/* Computation functions */
void FC_gemm_fp(int m, int n, int k, float * A, int lda,
        float * B, int ldb, float * C, int ldc, int threads, int max_threads, rntm_t * rntm);
void CONV_fp(int l, int K, int B, int H, int W, int KH, int KW, int C,
        float * I, float * IP, float * O, float * F, double * time, int threads, int max_threads, rntm_t * rntm);

/* Communication functions */

int main(int argc, char * argv []) {

    int rank, size, i, s, l;
    double alpha = 1.0, beta = 0.0;
    double time = 0.0; 
    
    if (argc < 4 ){
      perror("Usage: ./dnn model.csv steps teams num_threads\n");
      exit(-1);
    }

    int nsteps = atoi(argv[2]);
    int teams = atoi(argv[3]);
    int max_threads = (argv[4] == NULL) ? 1 : atoi(argv[4]);

    bli_init();
    rntm_t rntm;
    bli_rntm_init(&rntm);
    
    FILE *fp_model, *fp_results;
    int aux, j;
    char auxstr[200], auxstr2[200], *token, *str;
    
#ifdef PROGRESS
    printf("Model: %s\n", argv[1]);
#endif
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
#ifdef PROGRESS
      printf("layer %d, type %d, neurons %d, image_size %d, channels %d, kwidth %d, kheight %d, hstrides %d, vstrides %d,  procs %d\n",i,type[i],nneurons[i] ,image_size[i],channels[i],kwidth[i],kheight[i],hstrides[i],vstrides[i],procs[i]);
#endif      
	i++;
    }
    fclose(fp_model);

#ifdef TIMER
    double *step_timer = (double *) malloc(sizeof(double) * nsteps);
    double ** fp_comp_timer = (double **) malloc(sizeof(double) * nsteps); 
    double ** fp_im2col_timer = malloc(sizeof(double) * nsteps);
    double ** fp_comp_gflops = malloc(sizeof(double) * nsteps);
    double ** fp_comp_gflops_per_thread = malloc(sizeof(double) * nsteps);
    
    for (i = 0; i < nsteps; i++){
        fp_comp_timer[i] = (double *) malloc(sizeof (double) * NUM_LAYERS);
        fp_im2col_timer[i] = (double *) malloc(sizeof (double) * NUM_LAYERS);
        fp_comp_gflops[i] = (double *) malloc(sizeof (double) * NUM_LAYERS);
        fp_comp_gflops_per_thread[i] = (double *) malloc(sizeof (double) * NUM_LAYERS);
    }
#else
   
    double ** fp_im2col_timer = malloc(sizeof(double) * nsteps);

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
   /* float * matrix_A = malloc(max_size_fc * sizeof ( float));
    float * matrix_B = malloc(max_size_fc * sizeof ( float));
    float * matrix_C = malloc(max_size_fc * sizeof ( float));
*/
    size_t max_i = 0;//channels[0] * BATCH_SIZE * image_size[0] * image_size[0];
    size_t max_ip = 0;//channels[0] * kwidth[0] * kheight[0] * BATCH_SIZE * image_size[0] * image_size[0];
    size_t max_o = 0;//nkernels[0] * BATCH_SIZE * image_size[0] * image_size[0];
    size_t max_f = 0;//nkernels[0] * channels[0] * kwidth[0] * kheight[0];
    for (l = 1; l < NUM_LAYERS; l++) {
        if (type[l] == CONV) {
            size_t mi = channels[l-1] * BATCH_SIZE * image_size[l-1] * image_size[l-1];
            size_t mip = channels[l-1] * kwidth[l] * kheight[l] * BATCH_SIZE * image_size[l-1] * image_size[l-1];
            size_t mo = nkernels[l] * BATCH_SIZE * image_size[l-1] * image_size[l-1];
            size_t mf = nkernels[l] * channels[l-1] * kwidth[l] * kheight[l];
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
#ifdef PROGRESS
    printf("mi = %lu | mip = %lu | mo = %lu | mf = %lu\n", max_i, max_ip, max_o, max_f);
#endif
    /*float * conv_i = malloc(max_i * sizeof (float));
    float * conv_ip = malloc(max_ip * sizeof (float));
    float * conv_o = malloc(max_o * sizeof (float));
    float * conv_f = malloc(max_f * sizeof (float));*/ 
    /* The maximum size of the matrix model fits with the one of the matrix */
    size_t model_size = (max_size_fc > max_size_conv) ? max_size_fc : max_size_conv; //This is the maximum size of the model layers
    //float * model = malloc(model_size * sizeof (float));
    /* The data size is equal to the neurons in the first layer per the batch size */
    size_t data_size = nneurons[0] * BATCH_SIZE;
    //float * data = malloc(data_size * sizeof (float));




    /* Warm-up zone */
    int omp_warm;
#pragma omp parallel
    {
        omp_warm = omp_get_num_threads();
    }
    /*cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
            200, 200, 200, 1,
            matrix_A, 200, matrix_B, 200, 0, matrix_C, 200);
    */   
     int threads = omp_warm/teams;

   // printf("Tengo %d threads repartidos en %d teams de %d threads\n",omp_warm,teams,threads);
        time = omp_get_wtime();
        #pragma omp parallel num_threads(teams)
	{
    //	printf("Thread %d con team de %d threads reservando memoria...\n",omp_get_thread_num(), threads);
	
    	float * conv_i = malloc(max_i * sizeof (float));
    	float * conv_ip = malloc(max_ip * sizeof (float));
    	float * conv_o = malloc(max_o * sizeof (float));
    	float * conv_f = malloc(max_f * sizeof (float)); 
        
    	float * matrix_A = malloc(max_size_fc * sizeof ( float));
    	float * matrix_B = malloc(max_size_fc * sizeof ( float));
    	float * matrix_C = malloc(max_size_fc * sizeof ( float));
	
	#pragma omp for private(l)
        for (s = 0; s < nsteps; s++) {
#ifdef PROGRESS
            printf("Starting Step %d\n", s);
#endif
#ifdef TIMER
            step_timer[s] = omp_get_wtime();
#endif
            //Forward pass
            for (l = 1; l < NUM_LAYERS; l++) {
    //	printf("Thread %d em step %d layer %d...\n",omp_get_thread_num(),s,l);
#ifdef PROGRESS
                printf("FP layer %d ",l);
#endif
                        printf("Layer %d ", l);
                        if (type[l] == FC) { //FC
#ifdef PROGRESS
                		printf("FC \n");
#endif
                		printf("FC \n");
                            int m = nneurons[l]; //nneurons[l]/procs[l];//antes /size
                            int n = BATCH_SIZE;
                            int k = nneurons[l - 1]; //We need to reshape if the previous one was CONV
                            int lda = m;
                            int ldb = k;
                            int ldc = m;
#ifdef TIMER
                            fp_comp_timer[s][l] = omp_get_wtime();
#endif
                            FC_gemm_fp(m, n, k, matrix_A, lda, matrix_B, ldb, matrix_C, ldc, threads, max_threads, &rntm);
#ifdef TIMER
                            fp_comp_timer[s][l] = omp_get_wtime() - fp_comp_timer[s][l];
                            m = nneurons[l];
                            fp_comp_gflops[s][l] = (2.0 * m * n * k / fp_comp_timer[s][l]) / (1.0e+9);
                            fp_comp_gflops_per_thread[s][l] = fp_comp_gflops[s][l] / (1.0 * OMP_NUM_THREADS);
#endif

                        } else { //conv
               		    if(type[l] == CONV){
#ifdef PROGRESS
				printf("CONV \n");
#endif
				printf("CONV \n");
                                int num_kernels = nkernels[l]; //nneurons[l]/procs[l];//antes /size
                                int b = BATCH_SIZE;
                                int h = image_size[l - 1];
                                int w = image_size[l - 1];
                                int c = channels[l - 1];
                                int kh = kheight[l];
                                int kw = kwidth[l];
#ifdef TIMER
                                fp_comp_timer[s][l] = omp_get_wtime();
#endif
#ifdef PROGRESS
                                printf("%d, %d, %d, %d, %d,%d, %d\n",num_kernels, b, h, w, kh, kw, c);
				size_t aux = 1;
				printf("sizes mi = %lu | mip = %lu | mo = %lu | mf = %lu\n", c*b*h*w*aux, c*kw*kh*b*h*w*aux, num_kernels*b*h*w*aux, num_kernels*c*kw*kh*aux);
#endif
                                CONV_fp(l, num_kernels, b, h, w, kh, kw, c,
                                    conv_i, conv_ip, conv_o, conv_f, &fp_im2col_timer[s][l],threads,max_threads, &rntm);

#ifdef TIMER
                                fp_comp_timer[s][l] = omp_get_wtime() - fp_comp_timer[s][l];
                                int m = nkernels[l];
                                int n = b * h * w;
                                int k = c * kh *kw;
                                fp_comp_gflops[s][l] = (2.0 * m * n * k / fp_comp_timer[s][l]) / (1.0e+9);
                                fp_comp_gflops_per_thread[s][l] = fp_comp_gflops[s][l] / (1.0 * OMP_NUM_THREADS);
#endif
                           }
			   else{
#ifdef PROGRESS
				printf("OTRA \n");
#endif
#ifdef TIMER
                                fp_comp_timer[s][l] = 0.0;
                                fp_comp_gflops[s][l] = 0.0;
                                fp_comp_gflops_per_thread[s][l] = 0.0;
#endif

			   }
			}
                    }

#ifdef TIMER
            step_timer[s] = omp_get_wtime() - step_timer[s];
#endif
        } //steps
	} //parallel
        time = omp_get_wtime() - time;
#ifdef TIMER
#ifndef SUMMARY
            double total_time = 0.0;
            for (s = 0; s < nsteps; s++) {
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
            printf("Time per step = %f\n", total_time / nsteps);
#else

            double  total_time_r[NUM_LAYERS], total_time_fp[NUM_LAYERS], total_time_comp_fp[NUM_LAYERS]; 
            
            for (l = 1; l < NUM_LAYERS; l++) {
                total_time_r[l] = 0;
                total_time_fp[l] = 0;
                total_time_comp_fp[l] = 0;

            }
            for (s = 1; s < nsteps; s++) {
            for (l = 1; l < NUM_LAYERS; l++) {
                    total_time_fp[l] += fp_comp_timer[s][l];
                    total_time_comp_fp[l] += fp_comp_timer[s][l];
}
                    total_time_r[l] += total_time_fp[l];
                }
            double tt = 0.0;
            printf("#layer #threads total_time \n");
            for (l = 1; l < NUM_LAYERS; l++) {
		tt+=total_time_fp[l];
                printf("%d %d %f\n", l, OMP_NUM_THREADS, total_time_fp[l] / (nsteps - 1)); 
            }
            printf("Total %f \n", tt/ (nsteps - 1));

#endif
#endif    

    printf("Total %d steps, batches %d, teams %d => time %f (s)\n", nsteps, BATCH_SIZE, teams, time);

    /*free(matrix_A);
    free(matrix_B);
    free(matrix_C);*/
    //   free(model);
    //  free(data);
    return 0;
}

void FC_gemm_fp(int m, int n, int k, float * A, int lda, float * B, int ldb, float * C, int ldc, int threads, int max_threads, rntm_t * rntm) {
    //mkl_domain_set_num_threads(threads, MKL_DOMAIN_BLAS);
//	printf("FC con %d threads\n",threads);
    //omp_set_num_threads(threads);
  /*  #pragma omp parallel
{
	printf("Thread %d/%d En gemm \n",omp_get_thread_num(),omp_get_num_threads());
}*/
  //  printf("%d,%d,%d\n",m,n,k);
#ifndef NOGEMM  
  /*cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
          m, n, k, 1,
           A, lda, B, ldb, 0, C, ldc);*/
    test_gemm(m,n,k,max_threads,rntm);
#endif
    //printf("FP  FC  GEMM m(%d) : n(%d) : k(%d)\n", m, n, k);
}


void CONV_fp(int l, int K, int B, int H, int W, int KH, int KW, int C, float * I, float * IP, float * O, float * F, double * time, int threads, int max_threads, rntm_t * rntm) {

    // B batch size
    // Input image of size H x W, with C channels
    // K Kernels of size KH x KW each, with C channels
    /*
    float I[C][B][H][W];          // Input: C x (B H W)
    float IP[C][KW][KH][B][H][W]; // Patched input: (C K_H K_W) x (B H W)
    float O[K][B][H][W];          // Output: K x (B H W)
    float F[K][C][KH][KW];        // Filter: K x (C K_H K_W)
     */

//	printf("Conv con %d threads\n",threads);
    int b, h, w, kh, kw, c;
#ifdef TIMER
    *time = omp_get_wtime();
#endif
    // Im2col: I -> IP

    int kk1 = KH * KW * B * H*W;
    int  kk2 = KW * B * H*W;
    int  kk3 = B * H*W;
    int kk4 = H*W;
    int kk5 = B * (H + KH)*(W + KW);
    int kk6 = (H + KH)*(W + KW);
    int kk7 = (W + KW);
    int jk1, ik1, ik2, jk2, jk3, jk4, ik3, ik4, ik5;
//printf("Antes de im2col\n");
#ifndef NOIM2COL

#pragma omp parallel for private(b,h,w,kh,kw,ik1,ik2,ik3,ik4,ik5,jk1,jk2,jk3,jk4) num_threads(threads)
    for (c = 0; c < C; c++) {
//	printf("Thread %d/%d En im2Col it %d\n",omp_get_thread_num(),threads,c);
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
                        for (w = 0; w < W; w++){
                           IP[ ik5 + w ] = I[ /*jk4 +*/ w];
			}
                    }
                }
            }
        }
    }

#endif

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

    int m = K;
    int n = B * H*W;
    int k = C * KH*KW;
    int lda = m;
    int ldb = k;
    int ldc = m;
   // printf("%d,%d,%d\n",m,n,k);

    //mkl_domain_set_num_threads(threads, MKL_DOMAIN_BLAS);
    //omp_set_num_threads(threads);
/*    #pragma omp parallel
{
	printf("Thread %d/%d En gemm de im2col \n",omp_get_thread_num(),omp_get_num_threads());
}*/
#ifndef NOGEMM  
    /*cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
            m, n, k, 1,
            F, lda, IP, ldb, 0, O, ldc);*/
    test_gemm(m,n,k,max_threads,rntm);
#endif
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
