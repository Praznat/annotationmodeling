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
    int<lower=1> DIM_SIZE;

    int<lower=0> item_users[NITEMS, NUSERS];
    vector[DIM_SIZE] embeddings[NITEMS, NUSERS];
    int<lower=0, upper=NUSERS> n_gold_users; // first n_gold_users assumed to be gold
    real<lower=0> gold_uerr[NUSERS];

    # hyperparameters
    real<lower=0> uerr_prior_scale;
    real<lower=0> uerr_prior_loc_scale;
    real<lower=0> diff_prior_scale;
}
parameters {
    real<lower=0> uerr_center;
    vector<lower=0>[NUSERS] uerr;
    vector<lower=0>[NITEMS] diff;
    vector[DIM_SIZE] center[NITEMS];
}
transformed parameters {
    vector<lower=0>[NUSERS] uerr_g = uerr;
    if (n_gold_users > 0) {
        for (u in 1:NUSERS) {
            uerr_g[u] = gold_uerr[u];
        }
    }
}
model {
    uerr_center ~ normal(0, uerr_prior_loc_scale);
    diff ~ normal(0, diff_prior_scale);
    uerr ~ normal(uerr_center, uerr_prior_scale);
    // uerr ~ gamma(2, uerr_prior_scale);
    // diff ~ gamma(2, diff_prior_scale);

    for (i in 1:NITEMS) {
        for (u in 1:NUSERS) {
            real dist_from_center;
            int uid = item_users[i, u];
            if (uid == 0) break;
            dist_from_center = norm(embeddings[i, u] - center[i]);
            dist_from_center ~ normal(0, diff[i] + uerr_g[uid]);
        }
    }
}
generated quantities {
    vector[DIM_SIZE] item_user_errors[NITEMS, NUSERS] = embeddings;
    matrix<lower=0>[NITEMS, NUSERS] dist_from_truth = rep_matrix(666, NITEMS, NUSERS);

    for (i in 1:NITEMS) {
        for (u in 1:NUSERS) {
            real dist_from_center;
            int uid = item_users[i, u];
            if (uid == 0) break;
            dist_from_center = norm(embeddings[i, u] - center[i]);
            dist_from_truth[i, uid] = dist_from_center;
            // item_user_errors[i, uid] = embeddings[i, u] - center[i];
        }
    }
}