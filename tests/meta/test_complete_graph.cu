/************************************************************
 * File: graph_meta.cu
 *
 * Compile Example (需 C++11):
 *   nvcc graph_meta.cu -o graph_meta -Xcompiler -fopenmp --std=c++11
 *
 * Usage:
 *   ./graph_meta <graph_file> [dsatur_runs] [tabu_max_iter] [tabu_tenure]
 *
 * 改進:
 *  1) 修正 backupSol 未定義 => 用 backupColors.
 *  2) 分塊/抽樣：若衝突邊很多，隨機抽出一小部分 (chunkSize) 來嘗試，降低 CPU/GPU overhead.
 *  3) 保留 GPU 查找衝突邊/計算衝突、Tabu + Kempe + Aspiration、同一回合多Move 等。
 ************************************************************/

 #include <cstdio>
 #include <cstdlib>
 #include <cstring>
 #include <ctime>
 #include <omp.h>
 #include <vector>
 #include <algorithm>
 #include <tuple>
 #include <queue>    // BFS for Kempe chain
 #include <limits>
 #include <random>   // std::shuffle
 #include <cuda_runtime.h>
 
 //---------------------------------------------
 // 全域: CSR
 //---------------------------------------------
 static int* offsetArr    = nullptr; 
 static int* neighborsArr = nullptr;
 static int* degArr       = nullptr;
 
 //---------------------------------------------
 // GPU Kernel：找所有衝突邊
 //---------------------------------------------
 __global__ void kernelFindConflictEdges(
     const int* d_colors,
     const int* d_offset,
     const int* d_neighbors,
     int n,
     int* d_conflictCount,
     int2* d_conflictEdges  
 ){
     int v = blockDim.x * blockIdx.x + threadIdx.x;
     if(v >= n) return;
 
     int cv = d_colors[v];
     if(cv <= 0) return;
 
     int start = d_offset[v];
     int end   = d_offset[v+1];
 
     for(int e=start; e<end; e++){
         int w = d_neighbors[e];
         if(w>v && d_colors[w] == cv){
             int idx = atomicAdd(d_conflictCount, 1);
             d_conflictEdges[idx] = make_int2(v, w);
         }
     }
 }
 
 //---------------------------------------------
 // GPU Kernel：計算整體衝突數
 //---------------------------------------------
 __global__ void kernelCountConflictTotal(
     const int* d_colors,
     const int* d_offset,
     const int* d_neighbors,
     int n,
     int* d_conflictTotal
 ){
     int v = blockDim.x * blockIdx.x + threadIdx.x;
     if(v>=n) return;
 
     int cv = d_colors[v];
     if(cv<=0) return;
     
     int start = d_offset[v];
     int end   = d_offset[v+1];
     int localCount=0;
     for(int e=start; e<end; e++){
         int w = d_neighbors[e];
         if(w>v && d_colors[w] == cv){
             localCount++;
         }
     }
     atomicAdd(d_conflictTotal, localCount);
 }
 
 //---------------------------------------------
 // GPU 函式封裝
 //---------------------------------------------
 int gpuFindConflictEdges(
     int n,
     const int* colorsHost,
     const int* d_offset,
     const int* d_neighbors,
     int* d_colors,
     int2* d_conflictEdges,
     int* d_conflictCount,
     int2* conflictEdgesHost,
     int maxEdgeBufferSize
 ){
     // copy colors to GPU
     cudaMemcpy(d_colors, colorsHost, sizeof(int)*n, cudaMemcpyHostToDevice);
 
     // reset conflictCount
     int zero=0;
     cudaMemcpy(d_conflictCount, &zero, sizeof(int), cudaMemcpyHostToDevice);
 
     // kernel
     int blockSize=256;
     int gridSize=(n+blockSize-1)/blockSize;
     kernelFindConflictEdges<<<gridSize,blockSize>>>(
         d_colors, d_offset, d_neighbors, n, d_conflictCount, d_conflictEdges
     );
     cudaDeviceSynchronize();
 
     int conflictNum=0;
     cudaMemcpy(&conflictNum, d_conflictCount, sizeof(int), cudaMemcpyDeviceToHost);
     if(conflictNum>maxEdgeBufferSize){
         printf("**Warn** conflict edges exceed buffer size! %d / %d\n",
                conflictNum, maxEdgeBufferSize);
         conflictNum = maxEdgeBufferSize;
     }
 
     cudaMemcpy(conflictEdgesHost, d_conflictEdges, sizeof(int2)*conflictNum, cudaMemcpyDeviceToHost);
     return conflictNum;
 }
 
 int gpuCountConflictTotal(
     int n,
     const int* colorsHost,
     const int* d_offset,
     const int* d_neighbors,
     int* d_colors,
     int* d_conflictTotal
 ){
     // copy colors
     cudaMemcpy(d_colors, colorsHost, sizeof(int)*n, cudaMemcpyHostToDevice);
     // reset
     int zero=0;
     cudaMemcpy(d_conflictTotal, &zero, sizeof(int), cudaMemcpyHostToDevice);
 
     int blockSize=256;
     int gridSize=(n+blockSize-1)/blockSize;
     kernelCountConflictTotal<<<gridSize,blockSize>>>(
         d_colors, d_offset, d_neighbors, n, d_conflictTotal
     );
     cudaDeviceSynchronize();
 
     int ct=0;
     cudaMemcpy(&ct, d_conflictTotal, sizeof(int), cudaMemcpyDeviceToHost);
     return ct;
 }
 
 //---------------------------------------------
 // CPU: readFile
 //---------------------------------------------
 int readFile(const char* filename, int** vertices, char** edges, int* len){
     FILE* fp = fopen(filename, "r");
     if(!fp){
         printf("Error opening file: %s\n", filename);
         return -1;
     }
     printf("Reading file: %s\n", filename);
 
     char buffer[128];
     do {
         if(!fgets(buffer, 127, fp)){
             printf("Error reading file.\n");
             fclose(fp);
             return -1;
         }
     } while(buffer[0]=='c');  // skip comment
 
     if(buffer[0]=='p'){
         int edgesCount;
         sscanf(buffer+7, "%d %d", len, &edgesCount);
 
         *vertices = (int*) malloc((*len)*sizeof(int));
         *edges    = (char*)malloc((*len)*(*len)*sizeof(char));
         for(int i=0; i<(*len); i++){
             (*vertices)[i] = 0;
             for(int j=0; j<(*len); j++){
                 (*edges)[i*(*len)+j] = 0;
             }
         }
         while(!feof(fp)){
             if(!fgets(buffer,127,fp)) break;
             if(buffer[0]=='e'){
                 int s, t;
                 sscanf(buffer+2, "%d %d", &s, &t);
                 s--; t--;
                 (*edges)[ s*(*len)+t ] = 1;
                 (*edges)[ t*(*len)+s ] = 1;
             }
         }
     }
     else {
         printf("Wrong file format.\n");
         fclose(fp);
         return -1;
     }
     fclose(fp);
     return 0;
 }
 
 //---------------------------------------------
 // CPU: isColoringValidCSR
 //---------------------------------------------
 bool isColoringValidCSR(
     int n, 
     const int* offset, 
     const int* neighbors, 
     const int* colors
 ){
     int invalid=0;
     #pragma omp parallel for
     for(int v=0; v<n; v++){
         if(invalid) continue;
         int cv=colors[v];
         if(cv<=0){
             #pragma omp critical
             invalid=1;
         } else {
             int start=offset[v], end=offset[v+1];
             for(int e=start;e<end;e++){
                 int w=neighbors[e];
                 if(w>v && colors[w]==cv){
                     #pragma omp critical
                     invalid=1;
                     break;
                 }
             }
         }
     }
     return (invalid==0);
 }
 
 //---------------------------------------------
 // CPU: DSATUR
 //---------------------------------------------
 inline void setBit(std::vector<uint64_t> &bs, int idx){
     bs[idx>>6] |= (1ULL<<(idx&63));
 }
 inline bool testBit(const std::vector<uint64_t> &bs, int idx){
     return (bs[idx>>6] & (1ULL<<(idx&63))) != 0ULL;
 }
 
 int dsaturColoringCSR(
     int n, 
     const int* offset, 
     const int* neighbors,
     const int* deg,
     int* colors
 ){
     #pragma omp parallel for
     for(int i=0; i<n; i++){
         colors[i]=0;
     }
     std::vector<int> saturation(n,0);
     int coloredCount=0, usedColors=0;
 
     while(coloredCount<n){
         int maxSat=-1, candidate=-1;
 
         #pragma omp parallel
         {
             int localMax=-1, localCand=-1;
             #pragma omp for nowait
             for(int v=0; v<n; v++){
                 if(colors[v]==0){
                     if(saturation[v]>localMax){
                         localMax=saturation[v];
                         localCand=v;
                     } else if(saturation[v]==localMax && localCand!=-1){
                         if(deg[v]>deg[localCand]){
                             localCand=v;
                         }
                     }
                 }
             }
             #pragma omp critical
             {
                 if(localMax>maxSat){
                     maxSat=localMax;
                     candidate=localCand;
                 } else if(localMax==maxSat && localCand!=-1 && candidate!=-1){
                     if(deg[localCand]>deg[candidate]){
                         candidate=localCand;
                     }
                 }
             }
         }
 
         int bitSize=(usedColors+64)/64;
         std::vector<uint64_t> usedBits(bitSize,0ULL);
         int start=offset[candidate], end=offset[candidate+1];
         for(int e=start;e<end;e++){
             int nei=neighbors[e];
             int c=colors[nei];
             if(c>0 && c<=usedColors){
                 setBit(usedBits,c);
             }
         }
         int chosen=0;
         for(int c=1;c<=usedColors;c++){
             if(!testBit(usedBits,c)){
                 chosen=c;
                 break;
             }
         }
         if(!chosen){
             usedColors++;
             chosen=usedColors;
         }
         colors[candidate]=chosen;
         coloredCount++;
 
         #pragma omp parallel for
         for(int e2=start; e2<end; e2++){
             int nei=neighbors[e2];
             if(colors[nei]==0){
                 #pragma omp atomic
                 saturation[nei]++;
             }
         }
     }
     return usedColors;
 }
 
 int dsaturMultipleRunsCSR(
     int n, 
     const int* offset, 
     const int* neighbors, 
     const int* deg,
     int runs, 
     int* bestColors
 ){
     int bestUsed=n;
     std::vector<int> temp(n);
     for(int r=0; r<runs; r++){
         int used = dsaturColoringCSR(n, offset, neighbors, deg, temp.data());
         if(used<bestUsed){
             bestUsed=used;
             #pragma omp parallel for
             for(int i=0;i<n;i++){
                 bestColors[i]=temp[i];
             }
         }
     }
     return bestUsed;
 }
 
 //---------------------------------------------
 // CPU: Kempe chain swap
 //---------------------------------------------
 bool kempeChainSwap(
     int v, int w,
     int* colors,
     int n,
     const int* offset,
     const int* neighbors
 ){
     int cV = colors[v];
     int cW = colors[w];
     if(cV == cW) return false;
 
     std::vector<bool> visited(n,false);
     std::queue<int> Q;
     Q.push(v);
     visited[v]=true;
     std::vector<int> chain;
     chain.push_back(v);
 
     while(!Q.empty()){
         int x=Q.front(); Q.pop();
         int start=offset[x], end=offset[x+1];
         for(int e=start;e<end;e++){
             int nx=neighbors[e];
             if(!visited[nx] && (colors[nx]==cV || colors[nx]==cW)){
                 visited[nx]=true;
                 Q.push(nx);
                 chain.push_back(nx);
             }
         }
     }
 
     for(int node: chain){
         if(colors[node] == cV) colors[node] = cW;
         else if(colors[node]==cW) colors[node] = cV;
     }
     return true;
 }
 
 //---------------------------------------------
 // CPU: Tabu + Kempe + GPU => Advanced + chunk
 //---------------------------------------------
 int tabuSearchWithKempeAdvancedChunk(
     int n,
     const int* offset,
     const int* neighbors,
     int* colors,
     int maxIter,
     int tabuTenure,
     // GPU buffer
     int* d_offset,
     int* d_neighbors,
     int* d_colors,
     int2* d_conflictEdges,
     int* d_conflictCount,
     int* d_conflictTotal,
     int2* conflictEdgesHost,
     int maxEdgeBufferSize
 ){
     // 初始衝突
     int bestConflict = gpuCountConflictTotal(
         n, colors, d_offset, d_neighbors, d_colors, d_conflictTotal
     );
     std::vector<int> bestSol(n);
     #pragma omp parallel for
     for(int i=0;i<n;i++){
         bestSol[i] = colors[i];
     }
 
     // tabuList[v][c]
     std::vector< std::vector<int> > tabuList(n, std::vector<int>(n+1, 0));
 
     int iter=0;
     int bestIter=0;
 
     // 亂數引擎（用於隨機抽衝突邊）
     std::random_device rd;
     std::mt19937 g(rd());
 
     // 一次最多處理多少衝突邊
     const int chunkSize = 500;  // 可自行調
 
     while(iter < maxIter){
         if(bestConflict == 0) break;
         iter++;
 
         //=== (1) GPU 找衝突邊 ===
         int conflictNum = gpuFindConflictEdges(
             n, colors, d_offset, d_neighbors, 
             d_colors, d_conflictEdges, d_conflictCount,
             conflictEdgesHost, maxEdgeBufferSize
         );
 
         // 當前衝突
         int currConflict = gpuCountConflictTotal(
             n, colors, d_offset, d_neighbors, 
             d_colors, d_conflictTotal
         );
 
         //=== (2) 更新全局最佳 ===
         if(currConflict < bestConflict){
             bestConflict = currConflict;
             #pragma omp parallel for
             for(int i=0;i<n;i++){
                 bestSol[i] = colors[i];
             }
             bestIter = iter;
             if(bestConflict==0) break;
         }
         if(conflictNum<=0){
             // 無衝突邊 => conflict=0
             continue;
         }
 
         //=== (3) 抽樣衝突邊 (分塊 / chunk)===
         //    若 conflictNum > chunkSize，就隨機洗牌後取前 chunkSize
         //    以降低 GPU/CPU 大量嘗試 overhead
         std::vector<int2> edgesVec(conflictNum);
         for(int i=0;i<conflictNum;i++){
             edgesVec[i] = conflictEdgesHost[i];
         }
         std::shuffle(edgesVec.begin(), edgesVec.end(), g);
 
         if((int)edgesVec.size() > chunkSize){
             edgesVec.resize(chunkSize);
         }
 
         //=== (4) 產生鄰域 => 多個 move ===
         std::vector<int> backupColors(n);
         #pragma omp parallel for
         for(int i=0;i<n;i++){
             backupColors[i] = colors[i];
         }
 
         struct MoveCandidate {
             int v, newC;
             bool isKempe;
             int conflictVal;
             int w;
         };
 
         std::vector<MoveCandidate> candidateList;
         candidateList.reserve(edgesVec.size() * 3);
 
         #pragma omp parallel
         {
             std::vector<MoveCandidate> localMoves;
             std::vector<int> localBackup(n);
 
             #pragma omp for nowait
             for(int iEdge=0; iEdge<(int)edgesVec.size(); iEdge++){
                 int v = edgesVec[iEdge].x;
                 int w = edgesVec[iEdge].y;
                 int cV = backupColors[v];
                 int cW = backupColors[w];
                 if(cV<=0 || cW<=0) continue;
 
                 //---- a) v -> newColor
                 for(int c=1;c<=n;c++){
                     if(c==cV) continue;
                     bool isTabu=false;
                     if(tabuList[v][c]>iter){
                         isTabu=true;
                     }
                     // copy
                     for(int k=0;k<n;k++){
                         localBackup[k] = backupColors[k];
                     }
                     localBackup[v] = c;
 
                     int newConflict = gpuCountConflictTotal(
                         n, localBackup.data(),
                         d_offset, d_neighbors, d_colors, d_conflictTotal
                     );
                     // Aspiration
                     if(isTabu && newConflict>=bestConflict){
                         continue;
                     }
 
                     MoveCandidate mc;
                     mc.v = v;
                     mc.newC = c;
                     mc.isKempe=false;
                     mc.conflictVal=newConflict;
                     mc.w=-1;
                     localMoves.push_back(mc);
                 }
 
                 //---- b) w -> newColor
                 for(int c=1;c<=n;c++){
                     if(c==cW) continue;
                     bool isTabu=false;
                     if(tabuList[w][c]>iter){
                         isTabu=true;
                     }
                     for(int k=0;k<n;k++){
                         localBackup[k] = backupColors[k];
                     }
                     localBackup[w] = c;
 
                     int newConflict = gpuCountConflictTotal(
                         n, localBackup.data(),
                         d_offset, d_neighbors, d_colors, d_conflictTotal
                     );
                     if(isTabu && newConflict>=bestConflict){
                         continue;
                     }
                     MoveCandidate mc;
                     mc.v = w;
                     mc.newC = c;
                     mc.isKempe=false;
                     mc.conflictVal=newConflict;
                     mc.w=-1;
                     localMoves.push_back(mc);
                 }
 
                 //---- c) kempeChainSwap(v,w)
                 if(cV!=cW){
                     bool isTabu=false;
                     if(tabuList[v][cW]>iter && tabuList[w][cV]>iter){
                         // both tabu
                         isTabu=true;
                     }
                     // BFS swap
                     for(int k=0;k<n;k++){
                         localBackup[k] = backupColors[k];
                     }
                     // 做 swap
                     {
                         std::vector<bool> visited(n,false);
                         std::queue<int> Q;
                         Q.push(v);
                         visited[v]=true;
                         std::vector<int> chain;
                         chain.push_back(v);
 
                         while(!Q.empty()){
                             int x=Q.front(); Q.pop();
                             int start=offset[x], end=offset[x+1];
                             for(int e=start;e<end;e++){
                                 int nx=neighbors[e];
                                 int cX=localBackup[x];
                                 if(!visited[nx]){
                                     int cNX=localBackup[nx];
                                     if(cNX==cV || cNX==cW){
                                         visited[nx]=true;
                                         Q.push(nx);
                                         chain.push_back(nx);
                                     }
                                 }
                             }
                         }
                         for(int node: chain){
                             if(localBackup[node]==cV) localBackup[node]=cW;
                             else if(localBackup[node]==cW) localBackup[node]=cV;
                         }
                     }
                     int newConflict = gpuCountConflictTotal(
                         n, localBackup.data(),
                         d_offset, d_neighbors, d_colors, d_conflictTotal
                     );
                     if(isTabu && newConflict>=bestConflict){
                         // skip
                     } else {
                         MoveCandidate mc;
                         mc.v = v;
                         mc.newC=-1;
                         mc.isKempe=true;
                         mc.conflictVal=newConflict;
                         mc.w=w;
                         localMoves.push_back(mc);
                     }
                 }
             }
 
             #pragma omp critical
             {
                 candidateList.insert(candidateList.end(), localMoves.begin(), localMoves.end());
             }
         } // end parallel
 
         if(candidateList.empty()){
             continue;
         }
 
         //=== (5) 選衝突最小的 move ===
         std::sort(candidateList.begin(), candidateList.end(),
                   [](auto &a, auto &b){
                       return a.conflictVal < b.conflictVal;
                   });
         MoveCandidate bestMove = candidateList[0];
         int bestMoveConflict = bestMove.conflictVal;
 
         //--- (a) 還原 coloring: 用 backupColors
         #pragma omp parallel for
         for(int i=0;i<n;i++){
             colors[i] = backupColors[i];
         }
 
         //--- (b) 套用 bestMove
         if(!bestMove.isKempe){
             int oldC = backupColors[bestMove.v];
             if(oldC != bestMove.newC){
                 colors[bestMove.v] = bestMove.newC;
                 // update tabu
                 tabuList[bestMove.v][oldC] = iter + tabuTenure;
             }
         } else {
             // kempe swap(v,w)
             int v = bestMove.v;
             int w = bestMove.w;
             int cV = colors[v];
             int cW = colors[w];
             if(cV!=cW){
                 // BFS
                 std::vector<bool> visited(n,false);
                 std::queue<int> Q;
                 Q.push(v);
                 visited[v]=true;
                 std::vector<int> chain;
                 chain.push_back(v);
 
                 while(!Q.empty()){
                     int x=Q.front(); Q.pop();
                     int start=offset[x], end=offset[x+1];
                     for(int e=start;e<end;e++){
                         int nx=neighbors[e];
                         if(!visited[nx]){
                             if(colors[nx]==cV || colors[nx]==cW){
                                 visited[nx]=true;
                                 Q.push(nx);
                                 chain.push_back(nx);
                             }
                         }
                     }
                 }
                 for(int node: chain){
                     if(colors[node]==cV) colors[node]=cW;
                     else if(colors[node]==cW) colors[node]=cV;
                 }
                 // update tabu
                 tabuList[v][cV] = iter + tabuTenure;
                 tabuList[w][cW] = iter + tabuTenure;
             }
         }
     }
 
     // 最後用 bestSol
     for(int i=0;i<n;i++){
         colors[i] = bestSol[i];
     }
 
     // 回傳使用顏色數
     std::vector<bool> usedFlag(n+1,false);
     for(int i=0;i<n;i++){
         int c=colors[i];
         if(c>0 && c<=n) usedFlag[c]=true;
     }
     int usedCount=0;
     for(int c=1;c<=n;c++){
         if(usedFlag[c]) usedCount++;
     }
     return usedCount;
 }
 
 //---------------------------------------------
 // main
 //---------------------------------------------
 int main(int argc, char** argv){
     if(argc<2){
         printf("Usage: %s <graph_file> [dsatur_runs] [tabu_max_iter] [tabu_tenure]\n", argv[0]);
         return 1;
     }
     const char* filename=argv[1];
     int dsaturRuns=1;
     if(argc>=3) dsaturRuns=atoi(argv[2]);
     int maxIter=1000;
     if(argc>=4) maxIter=atoi(argv[3]);
     int tabuTenure=10;
     if(argc>=5) tabuTenure=atoi(argv[4]);
 
     //==== (1) 讀檔 & 建CSR ====
     int n;
     int* verts;
     char* mat;
     if(readFile(filename, &verts, &mat, &n)!=0){
         return -1;
     }
 
     degArr = (int*)malloc(sizeof(int)*n);
     offsetArr = (int*)malloc(sizeof(int)*(n+1));
 
     #pragma omp parallel for
     for(int v=0; v<n; v++){
         int count=0;
         for(int w=0; w<n; w++){
             if(mat[v*n+w]) count++;
         }
         degArr[v]=count;
     }
     offsetArr[0] = 0;
     for(int v=1; v<=n; v++){
         offsetArr[v] = offsetArr[v-1] + degArr[v-1];
     }
 
     neighborsArr = (int*)malloc(sizeof(int)*offsetArr[n]);
     #pragma omp parallel for
     for(int v=0; v<n; v++){
         int idx = offsetArr[v];
         for(int w=0; w<n; w++){
             if(mat[v*n+w]){
                 neighborsArr[idx++] = w;
             }
         }
     }
     free(verts);
     free(mat);
 
     //==== (2) DSATUR baseline (CPU) ====
     std::vector<int> dsaturColors(n,0);
     clock_t t1=clock();
     int dsaturUsed = dsaturMultipleRunsCSR(n, offsetArr, neighborsArr, degArr, dsaturRuns, dsaturColors.data());
     clock_t t2=clock();
     double dsaturSec = (double)(t2 - t1)/CLOCKS_PER_SEC;
     printf("DSATUR best of %d => used %d colors, time=%.3f\n", dsaturRuns, dsaturUsed, dsaturSec);
 
     if(!isColoringValidCSR(n, offsetArr, neighborsArr, dsaturColors.data())){
         printf("** DSATUR invalid??**\n");
     }
 
     //==== (3) 建 GPU buffer ====
     int *d_offset=nullptr, *d_neighbors=nullptr, *d_colors=nullptr;
     int *d_conflictCount=nullptr, *d_conflictTotal=nullptr;
     int2* d_conflictEdges=nullptr;
 
     cudaMalloc(&d_offset,    sizeof(int)*(n+1));
     cudaMalloc(&d_neighbors, sizeof(int)*offsetArr[n]);
     cudaMalloc(&d_colors,    sizeof(int)*n);
     cudaMalloc(&d_conflictCount, sizeof(int));
     cudaMalloc(&d_conflictTotal, sizeof(int));
 
     // 為示範起見，maxEdgeBufferSize = n*(n-1)/2 (完全圖)
     // 若圖非常大，實務上要考慮更好的方法以免記憶體爆炸
     size_t maxEdgeBufferSize = (size_t)n*(n-1)/2ULL;
     cudaMalloc(&d_conflictEdges, sizeof(int2)*maxEdgeBufferSize);
 
     cudaMemcpy(d_offset, offsetArr, sizeof(int)*(n+1), cudaMemcpyHostToDevice);
     cudaMemcpy(d_neighbors, neighborsArr, sizeof(int)*offsetArr[n], cudaMemcpyHostToDevice);
 
     // CPU buffer for conflictEdges
     std::vector<int2> conflictEdgesHost(maxEdgeBufferSize);
 
     //==== (4) Tabu + Kempe + chunk ====
     std::vector<int> tabuColors(dsaturColors);
     clock_t t3=clock();
     int tabuUsed = tabuSearchWithKempeAdvancedChunk(
         n, offsetArr, neighborsArr,
         tabuColors.data(),
         maxIter, tabuTenure,
         // GPU
         d_offset, d_neighbors, d_colors,
         d_conflictEdges, d_conflictCount, d_conflictTotal,
         conflictEdgesHost.data(),
         (int)maxEdgeBufferSize
     );
     clock_t t4=clock();
     double tabuSec = (double)(t4 - t3)/CLOCKS_PER_SEC;
 
     if(!isColoringValidCSR(n, offsetArr, neighborsArr, tabuColors.data())){
         printf("** Tabu+Kempe advanced chunk invalid??**\n");
     }
 
     printf("[Tabu+Kempe+Chunk] used %d colors, time=%.3f\n", tabuUsed, tabuSec);
 
     //==== (5) 比較結果 ====
     int finalUsed=dsaturUsed;
     const int* finalSol=dsaturColors.data();
     if(tabuUsed>0 && tabuUsed<dsaturUsed){
         finalUsed=tabuUsed;
         finalSol=tabuColors.data();
         printf("Tabu+Kempe chunk improved the solution.\n");
     } else {
         printf("No better solution than DSATUR.\n");
     }
 
     //==== (6) 輸出 ====
     FILE* fout = fopen("solution.sol","w");
     if(fout){
         for(int i=0; i<n; i++){
             fprintf(fout, "%d\n", finalSol[i]);
         }
         fclose(fout);
         printf("Final solution (k=%d) written to solution.sol\n", finalUsed);
     }
 
     //==== (7) 釋放 ====
     cudaFree(d_offset);
     cudaFree(d_neighbors);
     cudaFree(d_colors);
     cudaFree(d_conflictEdges);
     cudaFree(d_conflictCount);
     cudaFree(d_conflictTotal);
 
     free(degArr);
     free(offsetArr);
     free(neighborsArr);
 
     return 0;
 }
 
 
 
 
 
 
 