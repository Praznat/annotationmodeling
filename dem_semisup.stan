
data {
    // observed data
    int<lower=0> NDATA;
    int<lower=0> NITEMS;
    int<lower=0> NUSERS;
    int<lower=1, upper=NITEMS> items[NDATA];
    int<lower=1, upper=NUSERS> u1s[NDATA];
    int<lower=1, upper=NUSERS> u2s[NDATA];
    int<lower=0, upper=NUSERS> n_gold_users; // first n_gold_users assumed to be gold
    real<lower=0> gold_uerr[NUSERS];
    real distances[NDATA];

    // hyperparameters
    real<lower=0> uerr_prior_scale;
    real<lower=0> diff_prior_scale;

    real<lower=0> uerr_prior_loc_scale;
    real<lower=0> diff_prior_loc_scale;
}
transformed data {
    matrix[NUSERS, NUSERS] distance_matrices[NITEMS];
    vector[NUSERS] active_users_1hot[NITEMS];
    int active_users[NITEMS, NUSERS]; // for each item, user ids working on item until zero
    int n_itemlabels[NITEMS];
    for (i in 1:NITEMS) {
        distance_matrices[i] = rep_matrix(0, NUSERS, NUSERS);
        active_users_1hot[i] = rep_vector(0, NUSERS);
        for (u in 1:NUSERS) active_users[i, u] = 0;
    }
    for (n in 1:NDATA) {
        int i = items[n];
        int u1 = u1s[n];
        int u2 = u2s[n];
        real dist = distances[n];
        distance_matrices[i, u1, u2] = dist;
        distance_matrices[i, u2, u1] = dist;
        active_users_1hot[i, u1] = 1;
        active_users_1hot[i, u2] = 1;
    }
    for (i in 1:NITEMS) {
        int c = 1;
        for (u in 1:NUSERS) {
            if (active_users_1hot[i, u] > 0) {
                active_users[i, c] = u;
                c += 1;
            }
        }
        n_itemlabels[i] = c - 1;
    }
}
parameters {
    real<lower=0> uerr_prior_loc;
    // real<lower=0> diff_prior_loc;
    // real<lower=0> uerr_prior_scale;
    // real<lower=0> diff_prior_scale;
    vector<lower=0>[NUSERS] uerr;
    vector<lower=0>[NITEMS] diff;
}
transformed parameters {
    matrix[NITEMS, NUSERS] label_logprobs;
    matrix[NITEMS, NUSERS] label_logprobabilities = rep_matrix(-666, NITEMS, NUSERS);
    matrix[NITEMS, NUSERS] label_probabilities = rep_matrix(0, NITEMS, NUSERS);
    vector<lower=0>[NUSERS] uerr_g = uerr;
    if (n_gold_users > 0) {
        for (u in 1:NUSERS) {
            uerr_g[u] = gold_uerr[u];
        }
    }

    for (i in 1:NITEMS) {
        int nlabels = n_itemlabels[i];
        vector[nlabels] probs;
        if (nlabels == 0) continue;
        for (n1 in 1:nlabels) {
            real log_prob_n1 = 0;
            int u1 = active_users[i, n1];
            real scale = uerr_g[u1] * diff[i] + 0.01;
            for (n2 in 1:nlabels) {
                int u2 = active_users[i, n2];
                if (u1 != u2) {
                    real dist = distance_matrices[i, u1, u2];
                    log_prob_n1 += normal_lpdf(dist | 0, scale);
                }
            }
            label_logprobs[i, n1] = log_prob_n1;
        }
        probs = softmax(label_logprobs[i, 1:nlabels]');
        for (n in 1:nlabels) {
            int u = active_users[i, n];
            label_logprobabilities[i, u] = label_logprobs[i, n];
            label_probabilities[i, u] = probs[n];
        }
    }
}
model {

    for (u in 1:NUSERS) {
        uerr[u] ~ normal(uerr_prior_loc, uerr_prior_scale);
    }

    // diff ~ normal(diff_prior_loc, diff_prior_scale);
    uerr_prior_loc ~ gamma(2, 0.1);
    // uerr_prior_loc ~ normal(0, uerr_prior_loc_scale);
    // diff_prior_loc ~ normal(0, diff_prior_loc_scale);

    for (i in 1:NITEMS) {
        int nlabels = n_itemlabels[i];
        if (nlabels == 0) continue;
        target += log_sum_exp(label_logprobs[i, 1:nlabels]);
    }
}
generated quantities {
    matrix<lower=0>[NITEMS, NUSERS] dist_from_truth = 1 - label_probabilities;
    matrix[NUSERS, NUSERS] distance_matrices_data[NITEMS] = distance_matrices;
    int active_users_data[NITEMS, NUSERS] = active_users;
    int n_itemlabels_data[NITEMS] = n_itemlabels;
}
