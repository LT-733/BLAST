  trmv(unsigned int, char, const __half *, const __half *, __half *) (4, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         5.00
    SM Frequency                    Mhz       584.89
    Elapsed Cycles                cycle      202,738
    Memory Throughput                 %         5.35
    DRAM Throughput                   %         1.96
    Duration                         us       346.62
    L1/TEX Cache Throughput           %        91.55
    L2 Cache Throughput               %         0.38
    SM Active Cycles              cycle    11,863.27
    Compute (SM) Throughput           %         0.84
    ----------------------- ----------- ------------

    OPT   This kernel grid is too small to fill the available resources on this device, resulting in only 0.0 full      
          waves across all SMs. Look at Launch Statistics for more details.                                             

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                      4
    Registers Per Thread             register/thread              24
    Shared Memory Configuration Size           Kbyte           32.77
    Driver Shared Memory Per Block        byte/block               0
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              40
    Threads                                   thread           1,024
    Uses Green Context                                             0
    Waves Per SM                                                0.03
    -------------------------------- --------------- ---------------

    OPT   Est. Speedup: 90%                                                                                             
          The grid for this launch is configured to execute only 4 blocks, which is less than the GPU's 40              
          multiprocessors. This can underutilize some multiprocessors. If you do not intend to execute this kernel      
          concurrently with other workloads, consider reducing the block size to have at least one block per            
          multiprocessor or increase the size of the grid to fully utilize the available hardware resources. See the    
          Hardware Model (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model)            
          description for more details on launch configurations.                                                        

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block           10
    Block Limit Shared Mem                block           16
    Block Limit Warps                     block            4
    Theoretical Active Warps per SM        warp           32
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        22.47
    Achieved Active Warps Per SM           warp         7.19
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 77.53%                                                                                    
          The difference between calculated theoretical (100.0%) and measured achieved occupancy (22.5%) can be the     
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can   
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices   
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on     
          optimizing occupancy.                                                                                         

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle       33,889
    Total DRAM Elapsed Cycles        cycle   13,865,984
    Average L1 Active Cycles         cycle    11,863.27
    Total L1 Elapsed Cycles          cycle    8,125,976
    Average L2 Active Cycles         cycle    94,635.12
    Total L2 Elapsed Cycles          cycle    9,481,888
    Average SM Active Cycles         cycle    11,863.27
    Total SM Elapsed Cycles          cycle    8,125,976
    Average SMSP Active Cycles       cycle    11,384.17
    Total SMSP Elapsed Cycles        cycle   32,503,904
    -------------------------- ----------- ------------

    OPT   Est. Speedup: 5.497%                                                                                          
          One or more SMs have a much lower number of active cycles than the average number of active cycles. Maximum   
          instance value is 94.13% above the average, while the minimum instance value is 100.00% below the average.    
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 5.287%                                                                                          
          One or more SMSPs have a much lower number of active cycles than the average number of active cycles. Maximum 
          instance value is 94.35% above the average, while the minimum instance value is 100.00% below the average.    
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 5.497%                                                                                          
          One or more L1 Slices have a much lower number of active cycles than the average number of active cycles.     
          Maximum instance value is 94.13% above the average, while the minimum instance value is 100.00% below the     
          average.                                                                                                      

  trmv_optimized(unsigned int, char, const __half *, const __half *, __half *) (128, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         4.93
    SM Frequency                    Mhz       582.89
    Elapsed Cycles                cycle        7,823
    Memory Throughput                 %        40.48
    DRAM Throughput                   %        40.48
    Duration                         us        13.41
    L1/TEX Cache Throughput           %        40.65
    L2 Cache Throughput               %        10.33
    SM Active Cycles              cycle     5,037.82
    Compute (SM) Throughput           %        26.10
    ----------------------- ----------- ------------

    OPT   This kernel grid is too small to fill the available resources on this device, resulting in only 0.8 full      
          waves across all SMs. Look at Launch Statistics for more details.                                             

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                    128
    Registers Per Thread             register/thread              29
    Shared Memory Configuration Size           Kbyte           32.77
    Driver Shared Memory Per Block        byte/block               0
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              40
    Threads                                   thread          32,768
    Uses Green Context                                             0
    Waves Per SM                                                0.80
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block            8
    Block Limit Shared Mem                block           16
    Block Limit Warps                     block            4
    Theoretical Active Warps per SM        warp           32
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        65.93
    Achieved Active Warps Per SM           warp        21.10
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 34.07%                                                                                    
          The difference between calculated theoretical (100.0%) and measured achieved occupancy (65.9%) can be the     
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can   
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices   
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on     
          optimizing occupancy.                                                                                         

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle       26,733
    Total DRAM Elapsed Cycles        cycle      528,384
    Average L1 Active Cycles         cycle     5,037.82
    Total L1 Elapsed Cycles          cycle      313,872
    Average L2 Active Cycles         cycle     7,049.66
    Total L2 Elapsed Cycles          cycle      363,968
    Average SM Active Cycles         cycle     5,037.82
    Total SM Elapsed Cycles          cycle      313,872
    Average SMSP Active Cycles       cycle     4,962.52
    Total SMSP Elapsed Cycles        cycle    1,255,488
    -------------------------- ----------- ------------

    OPT   Est. Speedup: 15.43%                                                                                          
          One or more SMs have a much higher number of active cycles than the average number of active cycles.          
          Additionally, other SMs have a much lower number of active cycles than the average number of active cycles.   
          Maximum instance value is 24.03% above the average, while the minimum instance value is 22.11% below the      
          average.                                                                                                      
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 16.92%                                                                                          
          One or more SMSPs have a much lower number of active cycles than the average number of active cycles. Maximum 
          instance value is 26.75% above the average, while the minimum instance value is 29.91% below the average.     
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 15.43%                                                                                          
          One or more L1 Slices have a much higher number of active cycles than the average number of active cycles.    
          Additionally, other L1 Slices have a much lower number of active cycles than the average number of active     
          cycles. Maximum instance value is 24.03% above the average, while the minimum instance value is 22.11% below  
          the average.                                                                                                  
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 10.33%                                                                                          
          One or more L2 Slices have a much higher number of active cycles than the average number of active cycles.    
          Maximum instance value is 16.66% above the average, while the minimum instance value is 3.92% below the       
          average.                                                           