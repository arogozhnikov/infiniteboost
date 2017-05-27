module fortranfunctions
implicit none
contains
! this module contains functions that we need to work quite fast to perform research


! this function computes lambda-gradients in lambda-loss ndcg
subroutine compute_lambdas_fortran(n_docs, qid_indices, scores, normalized_gains, &
                                   & n_threads, orders, gradients, hessians)
    ! using openmp
    use omp_lib
    implicit none

    ! input
    integer*4, intent(in) :: n_docs, n_threads
    ! boundaries of queries
    integer*4, intent(in) :: qid_indices(:)
    real*4, intent(in)    :: scores(0:n_docs-1), normalized_gains(0:n_docs-1)
    ! orders contain corresponding queries for each position.
    ! passing this information allows avoiding complete resort on each iteration
    integer*4, intent(inout) :: orders(0:n_docs-1)
    real*4, intent(out)   :: gradients(0:n_docs-1), hessians(0:n_docs-1)


    integer*4 :: i, j, temp, query_id, query_start, query_end
    real*4 :: discounts(0:n_docs-1), lambda, sigmoid
    call omp_set_num_threads(n_threads)

    !$OMP PARALLEL DO private(i, j, temp, query_id, query_start, query_end, lambda, sigmoid) &
    !$OMP & SCHEDULE(DYNAMIC, 100)
    do query_id = lbound(qid_indices, 1), ubound(qid_indices, 1) - 1
        query_start = qid_indices(query_id)
        query_end = qid_indices(query_id + 1) - 1

        ! simple sorting, complexity proportional to number of permutations
        i = query_start + 1
        do while (i <= query_end )
            if (i == query_start) then
                i = i + 1
            else if (scores(orders(i)) <= scores(orders(i - 1))) then
                i = i + 1
            else
                temp = orders(i)
                orders(i) = orders(i - 1)
                orders(i - 1) = temp
                i = i - 1
            end if
        end do

        ! assigning necessary discounts
        do i = query_start, query_end
            discounts(orders(i)) = log(2.) / log(1. + i - query_start + 1)
        end do

        ! collecting gradients and hessians
        do i = query_start, query_end
            do j = i + 1, query_end
                sigmoid = 1 / (1. + exp((scores(j) - scores(i)) &
                & * sign(1., normalized_gains(j) - normalized_gains(i)) ))

                lambda = abs(discounts(j) - discounts(i)) &
                & * ( normalized_gains(i) - normalized_gains(j)) * sigmoid

                gradients(i) = gradients(i) + lambda
                gradients(j) = gradients(j) - lambda

                hessians(i)  = hessians(i) + abs(lambda) * (1 - sigmoid)
                hessians(j)  = hessians(j) + abs(lambda) * (1 - sigmoid)
            end do
        end do
    end do
end subroutine




subroutine build_decision_fortran(X, targets, weights, bootstrap_weights, current_indices, columns_to_test, n_thresh, &
    depth, n_current_leaves, reg, use_friedman_mse, n_threads, all_improvements)
    ! using openmp
    use omp_lib
    ! input
    integer*1, intent(in) :: X(:, :)
    real*4,    intent(in) :: targets(:), weights(:), bootstrap_weights(:)
    integer*4, intent(in) :: current_indices(:)
    integer*4, intent(in) :: columns_to_test(:)
    integer*4, intent(in) :: n_thresh, depth, n_current_leaves, use_friedman_mse, n_threads
    real*8,    intent(in) :: reg
    ! output
    real*8,    intent(out) :: all_improvements(n_current_leaves, size(columns_to_test), 0:n_thresh-1)

    real*4, allocatable:: bin_gradients_flat(:), bin_hessians_flat(:)
    real*8, allocatable:: bin_gradients(:, :), bin_hessians(:, :)

    real*8, dimension(2 ** (depth - 1)) :: temp, temp_gradients_op, temp_hessians_op
    real*4 :: gradients(size(targets)), hessians(size(targets))

    integer*4 :: leaf_indices(size(current_indices)), leaf, n_leaves, thresh, column_id, column, i
    call omp_set_num_threads(n_threads)

    n_leaves = 2 ** (depth - 1)

    !$OMP PARALLEL DO SCHEDULE(STATIC)
    do i = lbound(targets, 1), ubound(targets, 1)
        hessians(i) = weights(i) * bootstrap_weights(i)
        gradients(i) = targets(i) * hessians(i)
        leaf_indices(i) = iand(current_indices(i), n_leaves - 1)
    end do
    !$OMP END PARALLEL DO

    !$OMP PARALLEL DO private(bin_gradients_flat, bin_hessians_flat, bin_gradients, bin_hessians, thresh, &
    !$OMP &                   i, temp, column, leaf, temp_gradients_op, temp_hessians_op) &
    !$OMP & SCHEDULE(DYNAMIC, 1)
    do column_id = lbound(columns_to_test, 1), ubound(columns_to_test, 1)
        column = columns_to_test(column_id) + 1

        allocate(bin_gradients_flat(0:n_thresh * n_leaves - 1))
        allocate(bin_hessians_flat(0:n_thresh * n_leaves - 1))
        allocate(bin_gradients(0:n_leaves - 1, 0:n_thresh - 1))
        allocate(bin_hessians(0:n_leaves - 1, 0:n_thresh - 1))

        bin_gradients_flat(:) = 0
        bin_hessians_flat(:) = 0

        !$OMP SIMD
        do i = 1, size(leaf_indices, 1)
            leaf = IOR(leaf_indices(i), lshift(int(X(i, column), 4), depth - 1))
            bin_gradients_flat(leaf) = bin_gradients_flat(leaf) + gradients(i)
            bin_hessians_flat(leaf)  = bin_hessians_flat(leaf) + weights(i)
        end do

        bin_gradients(:, :) = reshape(bin_gradients_flat, shape(bin_gradients))
        bin_hessians(:, :) = reshape(bin_hessians_flat, shape(bin_hessians))

        do thresh = 1, n_thresh - 1
            bin_gradients(:, thresh) = bin_gradients(:, thresh) + bin_gradients(:, thresh - 1)
            bin_hessians(:, thresh)  = bin_hessians(:, thresh) + bin_hessians(:, thresh - 1)
        end do

        do thresh = 0, n_thresh - 1
            temp_gradients_op(:) = bin_gradients(:, n_thresh - 1) - bin_gradients(:, thresh)
            temp_hessians_op(:) = bin_hessians(:, n_thresh - 1) - bin_hessians(:, thresh)

            if (use_friedman_mse == 0) then
                temp(:) = bin_gradients(:, thresh) ** 2 / (bin_hessians(:, thresh) + reg) + &
                    temp_gradients_op(:) ** 2 / (temp_hessians_op(:) + reg)
            else
                temp(:) = (bin_gradients(:, thresh) * temp_hessians_op(:) - temp_gradients_op(:) * bin_hessians(:, thresh)) ** 2.
                temp(:) = temp(:) / &
                    ((bin_hessians(:, thresh) + reg) * (temp_hessians_op(:) + reg) * (bin_hessians(:, n_thresh - 1) + reg))
            end if
            all_improvements(:, column_id, thresh) = temp(:)
        end do

        deallocate(bin_gradients_flat, bin_hessians_flat, bin_gradients, bin_hessians)
    end do
    !$OMP END PARALLEL DO

end subroutine

end module


