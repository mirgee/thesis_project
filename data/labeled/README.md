corr    m       tau     w_t
lyap    m       tau     w_t
higu    w_w     w_s     num_k   k_max
sampen  m       tol

m ... embedding dimension
tau ... time delay
w_t ... Theiler window (minimum distance between neighbors)
w_w ... window width
w_s ... window slide
num_k ... number of k arrays ("lines") to generate
k_max ... maximum value of k (k's are generated on logscale from 2 to k_max), i.e. maximum length of a "line"
tol ... distance threshold for two vectors be considered equal (e.g. 20std means 0.2*std(data))
