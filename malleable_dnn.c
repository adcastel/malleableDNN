#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <math.h>
#include "omp.h"
//#include "cblas.h"
//#include "mkl.h"
#include "blis.h"
#ifdef TRAZA
#include "extrae_user_events.h"
#endif


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


#ifdef DOUBLE
#define DATA_TYPE double
#define B_DATA_TYPE BLIS_DOUBLE
#else
#define DATA_TYPE float
#define B_DATA_TYPE BLIS_FLOAT
#endif

/* Model features */

//#define NUM_STEPS  3  // Steps of the simulation
//#define BATCH_SIZE  64 // Batch size

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
    dt_a = B_DATA_TYPE;    
    dt_b = B_DATA_TYPE;    
    dt_c = B_DATA_TYPE;    
    dt_alpha = B_DATA_TYPE;
    dt_beta = B_DATA_TYPE; 

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


/* Computation functions */
#ifdef TESTGEMM
void FC_gemm_fp(int m, int n, int k, DATA_TYPE * A, int lda, DATA_TYPE * B, int ldb, DATA_TYPE * C, int ldc, int threads, int max_threads, rntm_t * rntm) {
    test_gemm(m,n,k,max_threads,rntm);
}
#else
void FC_gemm_fp(obj_t * a, obj_t *b, obj_t *c, obj_t * alpha, obj_t * beta ,rntm_t * rntm){
#ifndef NOGEMM  
    bli_gemm_ex(alpha, a, b, beta, c, NULL, rntm);
#endif
}
#endif


void CONV_fp(int l, int K, int B, int H, int W, int KH, int KW, int C, DATA_TYPE * I, 
        DATA_TYPE * IP, obj_t *a, obj_t * F, obj_t * O, obj_t * alpha, obj_t * beta, 
        double * time, int threads, int max_threads, rntm_t * rntm/*, int * current, int teams*/) {

    // B batch size
    // Input image of size H x W, with C channels
    // K Kernels of size KH x KW each, with C channels
    /*
    DATA_TYPE I[C][B][H][W];          // Input: C x (B H W)
    DATA_TYPE IP[C][KW][KH][B][H][W]; // Patched input: (C K_H K_W) x (B H W)
    DATA_TYPE O[K][B][H][W];          // Output: K x (B H W)
    DATA_TYPE F[K][C][KH][KW];        // Filter: K x (C K_H K_W)
     */
    int id = omp_get_thread_num();
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
#ifndef NOIM2COL
        
	int active = (threads > 4)? 4 : threads;
/*	int prev_curr = current[(id+1)%teams]
	current[id] = active;
        bli_rntm_set_active_ways(1,1,max,1,1,&rntm[(id+1)%teams]);
        bli_rntm_set_active_ways(1,1,active,1,1,&rntm[(id);*/
#pragma omp parallel for private(b,h,w,kh,kw,ik1,ik2,ik3,ik4,ik5,jk1,jk2,jk3,jk4) num_threads(active)
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
                        for (w = 0; w < W; w++){
                           IP[ ik5 + w ] = I[jk4 + w];
			}
                    }
                }
            }
        }
    }

#endif

#ifdef TIMER
    *time = omp_get_wtime() - *time;
#endif
    // Gemm

    int m = K;
    int n = B * H*W;
    int k = C * KH*KW;
    int lda = m;
    int ldb = k;
    int ldc = m;

#ifndef NOGEMM  
    /*cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
            m, n, k, 1,
            F, lda, IP, ldb, 0, O, ldc);*/
    //test_gemm(m,n,k,max_threads,rntm);
    bli_gemm_ex(alpha, a, F, beta, O, NULL, rntm);
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


/* Communication functions */
void  init_batch_size(int * BATCH_SIZE,int nsteps,int BATCH_SIZE_V, int * max_batch){
	int i;
	srand (2017);
	*max_batch = BATCH_SIZE_V;
	for(i=0; i<nsteps; i++){
		//BATCH_SIZE[i] = (BATCH_SIZE_V > 0) ? BATCH_SIZE_V : (pow(64,rand() % 2)) ; // 1 o 64
		BATCH_SIZE[i] = (BATCH_SIZE_V > 0) ? BATCH_SIZE_V : ((rand() % 8)+1) ; //Entre 1 y 8
		if(BATCH_SIZE[i] > *max_batch) *max_batch = BATCH_SIZE[i];
	//	printf("BATCH_SIZE[%d] = %d\n", i, BATCH_SIZE[i]);
	} 
	//printf("max_batch = %d\n", *max_batch);

}

void set_delays(int n, double * delays, double value){
    int i;
    srand (2017);
    int min = 5;
    int max = 20;
    delays[0]=0.0;
    for(i=1;i<n;i++){
        delays[i]= delays[i-1] + ((value >= 0) ? value : ((rand() % (max - min + 1)) + min)/100.0);
        //printf("Delays[%d] = %f", i,delays[i]);
    }
}


int main(int argc, char * argv []) {

    int rank, size, i, s, l;
    double alpha = 1.0, beta = 0.0;
    
    if (argc < 4 ){
      perror("Usage: ./dnn model.csv steps teams [num_threads] [batch_size] [malleable] [change] [min] [max]\n");
      exit(-1);
    }

    int nsteps = atoi(argv[2]);
    int teams = atoi(argv[3]);
    int max_threads = (argv[4] == NULL) ? 1 : atoi(argv[4]);
    int max, min, max_batch = 0;
    int BATCH_SIZE_V = (argv[5] == NULL) ? 0 : atoi(argv[5]);// Batch size. if 0, random values
    int * BATCH_SIZE = malloc (sizeof(int)*nsteps);
    float delays = (argv[6] == NULL) ? 0.0 : atof(argv[6]);// 0 or 1

    init_batch_size(BATCH_SIZE,nsteps,BATCH_SIZE_V, &max_batch);

    int malleable = (argv[7] == NULL) ? 0 : atoi(argv[7]);// 0 or 1
    int change = (argv[8] == NULL) ? 0 : atoi(argv[8]);// 0 or 1
    
    if(malleable){
        /*teams = 2;*/
        min =(argv[9] == NULL) ? 1 : atoi(argv[9]);
        max = (argv[10] == NULL) ? max_threads-1 : atoi(argv[10]);;
        
    }
    else{
	max = ceil(max_threads/(teams*1.0));
	min = max;
    }

    printf("Model %s. Malleable %d. Change %d. Steps %d. Teams %d. Max threads %d. Batch size %s. Delay %f \n",
		argv[1],malleable,change,nsteps,teams,max_threads,(argv[5]==0)?"Rand":argv[5], delays);
    
    bli_init();
    
    rntm_t * rntm = malloc(sizeof(rntm_t)*teams);
    int * current = malloc(sizeof(int)*teams);
   //creo los objetos rntm con el numero maximo de hilos 
   for(int t = 0; t< teams; t++){
    	bli_rntm_init(&rntm[t]);
	//printf("creo rntm[%d] con %d hilos\n",t,/*(malleable == 0) ?*/ max_threads/*  : max*/);
	printf("creo rntm[%d] con %d hilos\n",t,/*(malleable == 0) ? max_threads  :*/ max);
   	bli_rntm_set_ways(1,1,/*(malleable == 0) ? max_threads  :*/ max ,1,1,&rntm[t]);
    } 
    
    double * time = malloc(sizeof(double)*teams); 
    FILE *fp_model, *fp_results;
    int aux, j;
    char auxstr[200], auxstr2[200], *token, *str;
    
    // EMPIEZA LA LECTURA DEL MODELO
#ifdef PROGRESS
    printf("Model: %s\n", argv[1]);
#endif
    fp_model= fopen(argv[1], "r");
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
    int minsfc = 512;
    int minsconv = 8;
    int number_fc = 0;
    int number_conv = 0;
    fp_model= fopen(argv[1], "r");
    fgets(line, MAX_LEN, fp_model);
    i = 0;
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
            number_fc++;
	  }
    else if ( !strcmp(typel, "conv") ){ 
    	type[i] = CONV;  min_size[i]= minsconv; nkernels[i]= channels[i];
        number_conv++;
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
    // ACABA LA LECTURA DEL MODELO

#ifdef TIMER
    double *step_timer = (double *) malloc(sizeof(double) * nsteps);
    double *delay = (double *) malloc(sizeof(double) * nsteps);
    double *waiting_time = (double *) malloc(sizeof(double) * nsteps);
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

    set_delays(nsteps,delay,delays);

#else
   
    double ** fp_im2col_timer = malloc(sizeof(double) * nsteps);

#endif
    /* Model */


    //const char* env = getenv("OMP_NUM_THREADS");
    //int OMP_NUM_THREADS = (env != NULL) ? atoi(env) : 1;


    
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

    size_t max_i = 0;
    size_t max_ip = 0;
    size_t max_o = 0;
    size_t max_f = 0;
    for (l = 1; l < NUM_LAYERS; l++) {
        if (type[l] == CONV) {
            size_t mi = channels[l-1] * max_batch * image_size[l-1] * image_size[l-1];
            size_t mip = channels[l-1] * kwidth[l] * kheight[l] * max_batch * image_size[l-1] * image_size[l-1];
            size_t mo = nkernels[l] * max_batch * image_size[l-1] * image_size[l-1];
            size_t mf = nkernels[l] * channels[l-1] * kwidth[l] * kheight[l];
            if (mi > max_i) {  max_i = mi; }
            if (mip > max_ip) { max_ip = mip; }
            if (mo > max_o) { max_o = mo; }
            if (mf > max_f) { max_f = mf;  }
        }
    }
    

        #pragma omp parallel num_threads(teams)
	{
	int id = omp_get_thread_num();
        int threads;
     	
	if(!malleable){
            threads = max_threads/teams;
            int extra = max_threads % teams;
            if (id < extra)
		threads++;
        }
        else{
	    // Si soy el id0 me pongo los maximos y sino, los minimos
            threads = (id == 0) ? max : min;
        }
        current[id] = threads;
        
	bli_rntm_set_ways(1,1,threads,1,1,&rntm[id]);
    	printf("Thread %d con team de %d threads reservando memoria...\n",id, threads);
	
        obj_t * a = malloc(sizeof(obj_t) * NUM_LAYERS);    
        obj_t * b = malloc(sizeof(obj_t) * NUM_LAYERS);    
        obj_t * c = malloc(sizeof(obj_t) * NUM_LAYERS);
        
        obj_t alpha, beta;
        num_t dt_a, dt_b, dt_c, dt_alpha, dt_beta;
    //dim_t m, n, k;
        dt_a = B_DATA_TYPE;    
        dt_b = B_DATA_TYPE;    
        dt_c = B_DATA_TYPE;    
        dt_alpha = B_DATA_TYPE;
        dt_beta = B_DATA_TYPE;
        bli_obj_create( dt_alpha, 1, 1, 0, 0, &alpha ); 
        bli_obj_create( dt_beta,  1, 1, 0, 0, &beta );  
        
        int o;
        for(o = 1; o < NUM_LAYERS; o++){
            dim_t m, n, k;
            if(type[o] == FC){
                m = nneurons[o]; //nneurons[l]/procs[l];//antes /size
                n = max_batch;
                k = nneurons[o - 1];
                
                             //We need to reshape if the previous one was CONV            
            }
            else{
                if(type[o] == CONV){
                    m = nkernels[o];
                    n = max_batch * image_size[o - 1]*image_size[o - 1];
                    k = channels[o - 1] * kheight[o] * kwidth[o];
                    
                    
                }
            }
#ifdef PROGRESS
		printf("ID %d Layer %d Type %s %dx%dx%d\n", id, o, (type[o] == FC) ? "FC" : "CONV", m,n,k);
#endif
                    bli_obj_create( dt_a, m, k, 0, 0, &a[o] );         
                    bli_obj_create( dt_b, k, n, 0, 0, &b[o] );         
                    bli_obj_create( dt_c, m, n, 0, 0, &c[o] );         
                                                
                    bli_randm( &a[o] );                                
                    bli_randm( &b[o] );                                
                    bli_randm( &c[o] );          
                                                
        }                                        
        bli_setsc(  (0.9/1.0), 0.2, &alpha );           
        bli_setsc(  (1.0/1.0), 0.0, &beta );            

        
        int steps_by_id=0;
    	DATA_TYPE * conv_i = malloc(max_i * sizeof (DATA_TYPE));
    	DATA_TYPE * conv_ip = malloc(max_ip * sizeof (DATA_TYPE));
	printf("ID %d All the memory is already allocated!\n",id);
        #pragma omp barrier 
	
        time[id] = omp_get_wtime();
	//EMPIEZA EL PROCESADO DE CADA STEP
	double start_time = bli_clock();
	#pragma omp for private(l) schedule(dynamic,1) nowait
        for (s = 0; s < nsteps; s++) {
#ifdef PROGRESS
            printf("ID %d Starting Step %d\n", id, s);
#endif
#ifdef TIMER
            step_timer[s] = bli_clock();
#endif
	    steps_by_id++;
            //Forward pass
            for (l = 1; l < NUM_LAYERS; l++) {
            if(malleable == 1 && change /*1*/ == l){
	//	el actual se pone el maximo porque va a pasar por la parte costosa
                bli_rntm_set_active_ways(1,1,max,1,1,&rntm[id]);
	        current[id] = max;
		int tt;
	// 	al resto se les pone a minimo
		for(tt = 1; tt < teams; tt++){
                    bli_rntm_set_active_ways(1,1,min,1,1,&rntm[(id+tt)%teams]);
		    current[(id+tt)%teams] = min;
	        }
	    }
#ifdef PROGRESS
                printf("ID %d FP layer %d ",id, l);
#endif
                        if (type[l] == FC) { //FC
#ifdef PROGRESS
                		printf("FC \n");
#endif
#ifdef TESTGEMMS

                            int m = nneurons[l]; //nneurons[l]/procs[l];//antes /size
                            int n = BATCH_SIZE[s];
                            int k = nneurons[l - 1]; //We need to reshape if the previous one was CONV
                            int lda = m;
                            int ldb = k;
                            int ldc = m;
#endif
#ifdef TIMER
                            fp_comp_timer[s][l] = omp_get_wtime();
#endif
#ifdef TESTGEMMS
			//ESTO SOLO ES PARA EL TEST, POR ESO EL RNTM ES GENERICO
                            FC_gemm_fp(m, n, k, matrix_A, lda, matrix_B, ldb, matrix_C, ldc, threads, max_threads, &rntm);
#else
                            int m = nneurons[l]; //nneurons[l]/procs[l];//antes /size
                            int n = BATCH_SIZE[s];
                            int k = nneurons[l - 1]; //We need to reshape if the previous one was CONV
			   obj_t va, vb, vc;
#ifdef PROGRESS

			    printf("ID %d GEMM %dx%dx%d\n",id, m,n,k);

#endif
				bli_acquire_mpart( 0, 0, m, k, &a[l], &va );
				bli_acquire_mpart( 0, 0, k, n, &b[l], &vb );
				bli_acquire_mpart( 0, 0, m, n, &c[l], &vc );
                            FC_gemm_fp(&va,&vb,&vc, &alpha, &beta, &rntm[id]);
#endif
#ifdef TIMER
                            fp_comp_timer[s][l] = omp_get_wtime() - fp_comp_timer[s][l];
                            
                            fp_comp_gflops[s][l] = (2.0 * nneurons[l] * BATCH_SIZE[s] * nneurons[l-1] / fp_comp_timer[s][l]) / (1.0e+9);
                            fp_comp_gflops_per_thread[s][l] = fp_comp_gflops[s][l] / (1.0 /* * OMP_NUM_THREADS*/);
#endif

                        } else { //conv
               		    if(type[l] == CONV){
#ifdef PROGRESS
				printf("CONV \n");
#endif
                                int num_kernels = nkernels[l]; //nneurons[l]/procs[l];//antes /size
                                int bs = BATCH_SIZE[s];
                                int h = image_size[l - 1];
                                int w = image_size[l - 1];
                                int ch = channels[l - 1];
                                int kh = kheight[l];
                                int kw = kwidth[l];
#ifdef TIMER
                                fp_comp_timer[s][l] = omp_get_wtime();
#endif
#ifdef PROGRESS
                                printf("ID %d %d, %d, %d, %d, %d,%d, %d\n",id, num_kernels, bs, h, w, kh, kw, ch);
				size_t aux = 1;
				printf("sizes mi = %lu | mip = %lu | mo = %lu | mf = %lu\n", ch*bs*h*w*aux, ch*kw*kh*bs*h*w*aux, num_kernels*bs*h*w*aux, num_kernels*ch*kw*kh*aux);
#endif
                    	int m = nkernels[l];
                    	int n = bs * image_size[l - 1]*image_size[l - 1];
                    	int k = channels[l - 1] * kheight[l] * kwidth[l];
			        obj_t va, vb, vc;
				
				bli_acquire_mpart( 0, 0, m, k, &a[l], &va );
				bli_acquire_mpart( 0, 0, k, n, &b[l], &vb );
				bli_acquire_mpart( 0, 0, m, n, &c[l], &vc );
                                CONV_fp(l, num_kernels, bs, h, w, kh, kw, ch,
                                    conv_i, conv_ip, &va, &vb, &vc, 
                                        &alpha, &beta, &fp_im2col_timer[s][l],
                                        threads,max_threads, &rntm[id]);

#ifdef TIMER
                                fp_comp_timer[s][l] = omp_get_wtime() - fp_comp_timer[s][l];
                                //int m = nkernels[l];
                                int nn = bs * h * w;
                                int kk = ch * kh *kw;
                                fp_comp_gflops[s][l] = (2.0 * nkernels[l] * nn * kk / fp_comp_timer[s][l]) / (1.0e+9);
                                fp_comp_gflops_per_thread[s][l] = fp_comp_gflops[s][l] / (1.0 /* * OMP_NUM_THREADS*/);
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
            step_timer[s] = bli_clock() - step_timer[s];
            waiting_time[s] = (bli_clock() - (start_time + delay[s]))-step_timer[s];
#ifdef PROGRESS
	    printf("Time step %d = %f\n",s,step_timer[s]);
#endif
#endif
        } //steps
        time[id] = omp_get_wtime() - time[id];
    	printf("ID %d, Total %d steps, batches %d, teams %d => time %f (s)\n", id, steps_by_id, BATCH_SIZE_V, teams, time[id]);
	
    	FILE *fp_results;
	char buffer[32];
       snprintf(buffer, sizeof(char) * 32, "res_team%d.dat", id);

	fp_results= fopen(buffer,"w");
	for(int h = 1; h < NUM_LAYERS; h++){
		//fprintf(fp_results, "%s", bli_printm("matrix 'a1'", &c[h],"%5.1f",""));
		bli_fprintm(fp_results,"matrix 'a1'", &c[h],"%5.1f","");
	}
	fclose(fp_results);
	} //parallel
#ifdef TIMER
            double total_wt =0.0;
#ifndef SUMMARY
            double total_time = 0.0;
            for (s = 0; s < nsteps; s++) {
                printf("STEP %d\n Batch %d Time %f, waiting_time %f\n", s, BATCH_SIZE[s],step_timer[s],waiting_time[s]);
                total_wt+=waiting_time[s];
                printf("\t **** FP ****\n");
                for (l = 1; l < NUM_LAYERS; l++) {
		    if(type[l] != CONV && type[l] != FC) continue; 
                    printf("\t Layer %d (type %s)\n", l, (type[l] == CONV) ? "Conv" : "FC");
                    printf("\t\t FP Computation time %f", fp_comp_timer[s][l]);
                    if (type[l] == CONV) printf(" (im2col = %f)", fp_im2col_timer[s][l]);
                    printf(" | GFlops %f (cores %d) GFlops/core %f (cores %d)\n", fp_comp_gflops[s][l], 1 /*OMP_NUM_THREADS*/, fp_comp_gflops_per_thread[s][l], 1);//OMP_NUM_THREADS);
                }
                total_time += step_timer[s];
            }
            printf("Time per step = %f\n", total_time / nsteps);
            printf("Avg waiting time = %f\n", total_wt / nsteps);
#else

            double  total_time_r[NUM_LAYERS], total_time_fp[NUM_LAYERS], total_time_comp_fp[NUM_LAYERS]; 
            for (l = 1; l < NUM_LAYERS; l++) {
                total_time_r[l] = 0;
                total_time_fp[l] = 0;
                total_time_comp_fp[l] = 0;

            }
            for (s = 1; s < nsteps; s++) {
                total_wt+=waiting_time[s];
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
                printf("%d %d %f\n", l, max_threads, total_time_fp[l] / (nsteps - 1)); 
            }
            printf("Total %f Waiting_time %f\n", tt/ (nsteps - 1), total_wt/(nsteps-1));

#endif
#endif    

    //printf("Total %d steps, batches %d, teams %d => time %f (s)\n", nsteps, BATCH_SIZE_V, teams, time[id]);

    return 0;
}



