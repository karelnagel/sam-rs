use tch::Tensor;
use std::ops::Add;

pub fn window_partition(x: &Tensor, window_size: i64) -> (Tensor, (i64, i64)) {
    let (b, h, w, c) = x.size4().unwrap();

    let pad_h = (window_size - h % window_size) % window_size;
    let pad_w = (window_size - w % window_size) % window_size;
    let mut x = x.shallow_clone();
    if pad_h > 0 || pad_w > 0 {
        x = x.f_pad(&[0, pad_h, 0, pad_w, 0, 0], 0.0, 0);
    }
    let (hp, wp) = (h + pad_h, w + pad_w);

    x = x.view(&[
        b,
        hp / window_size,
        window_size,
        wp / window_size,
        window_size,
        c,
    ]);
    let windows =
        x.permute(&[0, 1, 3, 2, 4, 5])
            .contiguous()
            .view(&[-1, window_size, window_size, c]);
    (windows, (hp, wp))
}

pub fn window_unpartition(
    windows: &Tensor,
    window_size: i64,
    pad_hw: (i64, i64),
    hw: (i64, i64),
) -> Tensor {
    let (hp, wp) = pad_hw;
    let (h, w) = hw;
    let b = windows.size()[0] / (hp * wp / window_size / window_size);
    let mut x = windows
        .view(&[
            b,
            hp / window_size,
            wp / window_size,
            window_size,
            window_size,
            -1,
        ])
        .permute(&[0, 1, 3, 2, 4, 5])
        .contiguous()
        .view(&[b, hp, wp, -1]);

    if hp > h || wp > w {
        x = x.narrow(1, 0, h).narrow(2, 0, w).contiguous();
    }
    x
}

pub fn get_rel_pos(q_size: i64, k_size: i64, rel_pos: &Tensor) -> Tensor {
    let max_rel_dist = 2 * i64::max(q_size, k_size) - 1;
    let rel_pos_resized = if rel_pos.size()[0] != max_rel_dist {
        // Interpolate rel pos.
        let rel_pos_reshaped = rel_pos
            .view(&[1, rel_pos.size()[0], -1])
            .permute(&[0, 2, 1]);
        let rel_pos_resized = tch::vision::interpolate(
            &rel_pos_reshaped,
            tch::nn::functional::InterpolateOptions::default()
                .size(vec![max_rel_dist])
                .mode(tch::nn::functional::InterpolateMode::Linear),
        );
        rel_pos_resized.view(&[-1, max_rel_dist]).permute(&[1, 0])
    } else {
        rel_pos.shallow_clone()
    };

    // Scale the coords with short length if shapes for q and k are different.
    let q_coords = Tensor::arange(q_size, (tch::Kind::Float, rel_pos.device())).unsqueeze(1)
        * f64::max(k_size as f64 / q_size as f64, 1.0);
    let k_coords = Tensor::arange(k_size, (tch::Kind::Float, rel_pos.device())).unsqueeze(0)
        * f64::max(q_size as f64 / k_size as f64, 1.0);
    let relative_coords =
        (q_coords - k_coords) + (k_size - 1) as f64 * f64::max(q_size as f64 / k_size as f64, 1.0);

    rel_pos_resized.index_select(0, &relative_coords.to_kind(tch::Kind::Int64))
}

pub fn add_decomposed_rel_pos(
    attn: &Tensor,
    q: &Tensor,
    rel_pos_h: &Tensor,
    rel_pos_w: &Tensor,
    q_size: (i64, i64),
    k_size: (i64, i64),
) -> Tensor {
    let (q_h, q_w) = q_size;
    let (k_h, k_w) = k_size;
    let rh = get_rel_pos(q_h, k_h, rel_pos_h);
    let rw = get_rel_pos(q_w, k_w, rel_pos_w);

    let (b, _, dim) = q.size3().unwrap();
    let r_q = q.view(&[b, q_h, q_w, dim]);
    let rel_h = r_q.einsum("bhwc,hkc->bhwk", &[&rh]);
    let rel_w = r_q.einsum("bhwc,wkc->bhwk", &[&rw]);

    let attn = attn
        .view(&[b, q_h, q_w, k_h, k_w])
        .add(&rel_h.unsqueeze(-2))
        .add(&rel_w.unsqueeze(-3))
        .view(&[b, q_h * q_w, k_h * k_w]);

    attn
}
