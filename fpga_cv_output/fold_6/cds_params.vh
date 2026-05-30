// Auto-generated CDS model parameters
// Do not edit — regenerate with parameter_export.py

// Model dimensions
localparam N_NODES       = 99;
localparam N_FEATURES    = 279;
localparam N_DISEASES    = 12;

// Disease class mapping (class -> contiguous offset)
// Class  2 -> offset 0
// Class  3 -> offset 1
// Class  4 -> offset 2
// Class  5 -> offset 3
// Class  6 -> offset 4
// Class  7 -> offset 5
// Class  8 -> offset 6
// Class  9 -> offset 7
// Class 10 -> offset 8
// Class 14 -> offset 9
// Class 15 -> offset 10
// Class 16 -> offset 11

// Fixed-point constants (Q s2.30)
localparam signed [31:0] ONE_FP       = 32'sh40000000;  // 1.0
localparam signed [31:0] THRESHOLD_FP = 32'sh0199999A;  // 0.025

// Node index mapping
// Node   0 = "root"
// Node   1 = "root|k2_f1"
// Node   2 = "root|k2_f2"
// Node   3 = "root|k6_f1"
// Node   4 = "root|k6_f2"
// Node   5 = "root|k9_f1"
// Node   6 = "root|k9_f2"
// Node   7 = "root|k12_f1"
// Node   8 = "root|k12_f2"
// Node   9 = "root|k14_f1"
// Node  10 = "root|k14_f2"
// Node  11 = "root|k29_f1"
// Node  12 = "root|k29_f2"
// Node  13 = "root|k40_f1"
// Node  14 = "root|k40_f2"
// Node  15 = "root|k56_f1"
// Node  16 = "root|k56_f2"
// Node  17 = "root|k65_f1"
// Node  18 = "root|k65_f2"
// Node  19 = "root|k140_f1"
// Node  20 = "root|k140_f2"
// Node  21 = "root|k161_f1"
// Node  22 = "root|k161_f2"
// Node  23 = "root|k162_f1"
// Node  24 = "root|k162_f2"
// Node  25 = "root|k168_f1"
// Node  26 = "root|k168_f2"
// Node  27 = "root|k169_f1"
// Node  28 = "root|k169_f2"
// Node  29 = "root|k171_f1"
// Node  30 = "root|k171_f2"
// Node  31 = "root|k172_f1"
// Node  32 = "root|k172_f2"
// Node  33 = "root|k177_f1"
// Node  34 = "root|k177_f2"
// Node  35 = "root|k178_f1"
// Node  36 = "root|k178_f2"
// Node  37 = "root|k187_f1"
// Node  38 = "root|k187_f2"
// Node  39 = "root|k188_f1"
// Node  40 = "root|k188_f2"
// Node  41 = "root|k190_f1"
// Node  42 = "root|k190_f2"
// Node  43 = "root|k197_f1"
// Node  44 = "root|k197_f2"
// Node  45 = "root|k198_f1"
// Node  46 = "root|k198_f2"
// Node  47 = "root|k205_f1"
// Node  48 = "root|k205_f2"
// Node  49 = "root|k207_f1"
// Node  50 = "root|k207_f2"
// Node  51 = "root|k208_f1"
// Node  52 = "root|k208_f2"
// Node  53 = "root|k217_f1"
// Node  54 = "root|k217_f2"
// Node  55 = "root|k218_f1"
// Node  56 = "root|k218_f2"
// Node  57 = "root|k227_f1"
// Node  58 = "root|k227_f2"
// Node  59 = "root|k228_f1"
// Node  60 = "root|k228_f2"
// Node  61 = "root|k232_f1"
// Node  62 = "root|k232_f2"
// Node  63 = "root|k236_f1"
// Node  64 = "root|k236_f2"
// Node  65 = "root|k237_f1"
// Node  66 = "root|k237_f2"
// Node  67 = "root|k238_f1"
// Node  68 = "root|k238_f2"
// Node  69 = "root|k246_f1"
// Node  70 = "root|k246_f2"
// Node  71 = "root|k247_f1"
// Node  72 = "root|k247_f2"
// Node  73 = "root|k248_f1"
// Node  74 = "root|k248_f2"
// Node  75 = "root|k251_f1"
// Node  76 = "root|k251_f2"
// Node  77 = "root|k252_f1"
// Node  78 = "root|k252_f2"
// Node  79 = "root|k257_f1"
// Node  80 = "root|k257_f2"
// Node  81 = "root|k258_f1"
// Node  82 = "root|k258_f2"
// Node  83 = "root|k261_f1"
// Node  84 = "root|k261_f2"
// Node  85 = "root|k262_f1"
// Node  86 = "root|k262_f2"
// Node  87 = "root|k267_f1"
// Node  88 = "root|k267_f2"
// Node  89 = "root|k268_f1"
// Node  90 = "root|k268_f2"
// Node  91 = "root|k271_f1"
// Node  92 = "root|k271_f2"
// Node  93 = "root|k276_f1"
// Node  94 = "root|k276_f2"
// Node  95 = "root|k277_f1"
// Node  96 = "root|k277_f2"
// Node  97 = "root|k278_f1"
// Node  98 = "root|k278_f2"
