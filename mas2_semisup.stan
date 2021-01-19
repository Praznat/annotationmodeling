functions {
    // no built-in vector norm in Stan?
    real norm(vector x) {
        return sqrt(sum(square(x)));
    }
}

data {
    # observed data
    int<lower=0> NDATA;
    int<lower=0> NITEMS;
    int<lower=0> NUSERS;
    int<lower=1, upper=NITEMS> items[NDATA];
    int<lower=1, upper=NUSERS> u1s[NDATA];
    int<lower=1, upper=NUSERS> u2s[NDATA];
    int<lower=0, upper=NUSERS> n_gold_users; // first n_gold_users assumed to be gold
    real<lower=0> gold_uerr[NUSERS];
    real distances[NDATA];

    # hyperparameters
    int<lower=1> DIM_SIZE;
    int<lower=0> eps_limit;
    real<lower=0> uerr_prior_scale;
    real<lower=0> diff_prior_scale;
}
parameters {
    vector<lower=0>[NUSERS] uerr;
    vector<lower=0>[NITEMS] diff;
    real<lower=0> sigma;
    real<lower=0> sigma2;

    vector<lower=-eps_limit, upper=eps_limit>[DIM_SIZE] item_user_errors_Z[NITEMS, NUSERS];
}
transformed parameters {
    real pred_distances[NDATA];
    matrix<lower=0>[NITEMS, NUSERS] dist_from_truth = rep_matrix(666, NITEMS, NUSERS);
    vector<lower=0>[NUSERS] uerr_g = uerr;
    if (n_gold_users > 0) {
        for (u in 1:NUSERS) {
            uerr_g[u] = gold_uerr[u];
        }
    }

    for (n in 1:NDATA) {
        int u1 = u1s[n];
        int u2 = u2s[n];
        int item = items[n];
        
        vector[DIM_SIZE] iueZ1 = item_user_errors_Z[item, u1];
        vector[DIM_SIZE] iueZ2 = item_user_errors_Z[item, u2];

        dist_from_truth[item, u1] = norm(iueZ1);
        dist_from_truth[item, u2] = norm(iueZ2);

        pred_distances[n] = norm(iueZ1 - iueZ2);
    }
}
model {
    sigma ~ exponential(1);
    uerr ~ normal(1, uerr_prior_scale);
    diff ~ normal(1, diff_prior_scale);

    for (i in 1:NITEMS) {
        for (u in 1:NUSERS) {
            if (dist_from_truth[i, u] != 666) {
                item_user_errors_Z[i, u] ~ normal(0, diff[i] * uerr_g[u]);
            }
        }
    }

    // likelihood
    distances ~ normal(pred_distances, sigma);
}
generated quantities {
    vector[DIM_SIZE] item_user_errors[NITEMS, NUSERS] = item_user_errors_Z;
}