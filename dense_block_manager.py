import torch
import MinkowskiEngine as ME


class DenseBlockManager:
    def __init__(self, x_seq_sparse_data, batch_size, spatial_shape=(100, 368), patch_shape=(50, 46)):
        self.x_seq_sparse_data = x_seq_sparse_data
        self.batch_size = batch_size
        self.spatial_shape = spatial_shape
        self.patch_shape = patch_shape
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        height, width = spatial_shape
        patch_h, patch_w = patch_shape
        if height % patch_h != 0 or width % patch_w != 0:
            raise ValueError(
                f"patch_shape={patch_shape} must evenly divide spatial_shape={spatial_shape}."
            )

        self.grid_rows = height // patch_h
        self.grid_cols = width // patch_w
        self.patches_per_sample = self.grid_rows * self.grid_cols

    def get_block_dense(self, block_idx, block_size):
        start_t = block_idx * block_size
        end_t = min(start_t + block_size, len(self.x_seq_sparse_data))
        actual_block_size = end_t - start_t

        dense_block = torch.zeros(
            (actual_block_size, self.batch_size, 1, *self.spatial_shape),
            device=self.device,
        )

        for i, t in enumerate(range(start_t, end_t)):
            b_coords, b_feats = self.x_seq_sparse_data[t]

            if len(b_feats) == 0:
                continue

            sp_tensor = ME.SparseTensor(features=b_feats, coordinates=b_coords, device=self.device)
            dense_t = sp_tensor.dense(shape=torch.Size([self.batch_size, 1, *self.spatial_shape]))[0]
            dense_block[i] = dense_t

        patch_h, patch_w = self.patch_shape
        patches = dense_block.unfold(3, patch_h, patch_h).unfold(4, patch_w, patch_w)
        patches = patches.permute(0, 1, 3, 4, 2, 5, 6).contiguous()
        patches = patches.view(actual_block_size, self.batch_size * self.patches_per_sample, 1, patch_h, patch_w)

        return patches

    def stitch_patches(self, patch_maps):
        patch_count, channels, patch_h, patch_w = patch_maps.shape
        expected_patches = self.batch_size * self.patches_per_sample
        if patch_count != expected_patches:
            raise ValueError(f"Expected {expected_patches} patches, got {patch_count}.")

        patch_maps = patch_maps.view(
            self.batch_size,
            self.grid_rows,
            self.grid_cols,
            channels,
            patch_h,
            patch_w,
        )
        patch_maps = patch_maps.permute(0, 3, 1, 4, 2, 5).contiguous()
        return patch_maps.view(
            self.batch_size,
            channels,
            self.grid_rows * patch_h,
            self.grid_cols * patch_w,
        )
